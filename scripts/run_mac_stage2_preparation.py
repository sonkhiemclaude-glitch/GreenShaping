"""Mac-only persistent preparation. Wait for storage; never contacts cloud.

All acquisitions are TRAIN/select. No automatic retries; every child has a finite
runtime and belongs to an owned process group. Main-paper readiness stays false.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from greenshaping.utils.evidence import atomic_json
from greenshaping.utils.storage_budget import check as check_storage_budget
from greenshaping.utils.evidence_backup import snapshot, verify
from greenshaping.utils.provenance import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--wait-storage-seconds',type=int,default=14400)
    a=p.parse_args()
    if not 0<=a.wait_storage_seconds<=14400:p.error('finite storage wait required')
    a.out.mkdir(parents=True,exist_ok=False)
    app=Path(__file__).resolve().parents[1]
    child=None;stages={};deadline=None
    def terminate(*_):raise SystemExit(143)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    def stage(name,argv,timeout):
        nonlocal child
        if time.time()>=deadline or shutil.disk_usage(a.out).free<3*1024**3:
            raise RuntimeError('deadline/storage gate; no new acquisition')
        check_storage_budget(a.out,max_bytes=512*1024**2,reserve_bytes=3*1024**3,upcoming_bytes=128*1024**2)
        if name!='regression':
            manifest=json.loads((a.data/'store/projection.json').read_text())
            if set(manifest['files'])!={'projection.npy','order.npy','train_feature_sorted.npy'}:
                raise ValueError('unexpected projected data inventory')
            actual={f:file_sha256(a.data/'store'/f) for f in manifest['files']}
            if actual!=manifest['files']:raise ValueError('data integrity failure before '+name)
            atomic_json(a.out/('data_verified_'+name+'.json'),actual)
        atomic_json(a.out/'status.json',{'stage':name,'cloud_used':False,'test_used':False,'deadline_epoch':deadline})
        with (a.out/(name+'.log')).open('wb') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=log,cwd=app,start_new_session=True,
                 env=dict(os.environ,PYTHONPATH=str(app/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
            result=child.wait(timeout=min(timeout,max(1,deadline-time.time())))
        child=None
        if name=='regression' and (a.out/'pytest_tmp').exists():shutil.rmtree(a.out/'pytest_tmp')
        stages[name]={'returncode':result};atomic_json(a.out/'stages.json',stages)
        sources={p.name:p for p in a.out.iterdir() if p.name not in ['backup_audit.json','restore_verification']}
        snap=snapshot(sources,a.out.parent/'stage2_backups',a.out.name+'_'+name)
        audit=verify(a.out.parent/'stage2_backups',a.out.name+'_'+name)
        atomic_json(a.out/('backup_'+name+'.json'),{'snapshot':snap,'verification':audit})
        if not snap['complete']:raise RuntimeError('stage evidence snapshot incomplete')
        if result:raise RuntimeError(name+' failed; evidence retained and no automatic replay')
    try:
        until=time.time()+a.wait_storage_seconds
        while shutil.disk_usage(a.out).free<4*1024**3:
            atomic_json(a.out/'status.json',{'stage':'waiting_storage','free_bytes':shutil.disk_usage(a.out).free,
                'required_bytes':4*1024**3,'wait_deadline_epoch':until,'cloud_used':False,'test_used':False})
            if time.time()>=until:raise TimeoutError('storage wait expired; no measurements launched')
            time.sleep(30)
        deadline=time.time()+10800
        stage('regression',[sys.executable,'-m','pytest','-q','--basetemp',str(a.out/'pytest_tmp')],300)
        stage('train',[sys.executable,'scripts/mac_stage2_calibration.py','--data',str(a.data),'--out',str(a.out/'train'),
                       '--N','20','--phase','train','--max-seconds','3600'],3700)
        train=json.loads((a.out/'train/summary.json').read_text())
        frozen=json.loads((a.out/'train/frozen_shortlist.json').read_text())
        if not train['all_baselines_passed'] or not any(n.startswith('dp_') for n in frozen['shortlist']) or not any(n.startswith('combo_') for n in frozen['shortlist']):
            raise RuntimeError('no functional TRAIN shortlist; no select dispatch')
        stage('select',[sys.executable,'scripts/mac_stage2_calibration.py','--data',str(a.data),'--out',str(a.out/'select'),
                       '--N','20','--phase','select','--frozen',str(a.out/'train/frozen_shortlist.json'),'--max-seconds','1800'],1900)
        select=json.loads((a.out/'select/summary.json').read_text())
        if not select['all_baselines_passed']:raise RuntimeError('SELECT baseline failure; no automatic replay')
        engineering_selection={}
        for prefix in ['dp_','combo_']:
            feasible=[(n,v) for n,v in select['per_configuration'].items() if n.startswith(prefix) and v['all_passed']]
            if not feasible:raise RuntimeError('no feasible frozen '+prefix+' candidate on SELECT')
            engineering_selection[prefix]=min(feasible,key=lambda pair:pair[1]['mean_framed_plaintext_bytes'])[0]
        atomic_json(a.out/'engineering_selection.json',{'selected':engineering_selection,'test_used':False,
            'rule':'Minimum framed plaintext cost among frozen candidates passing all SELECT functional trials',
            'privacy_selection_done':False,'rejected_candidates_retained':True})
        for model,name in [('mistral-small3.1:24b','mistral'),('qwen2.5:14b-instruct-q4_K_M','qwen14b')]:
            for phase in ['train','select']:
                argv=[sys.executable,'scripts/validate_higgs_answer_workflow.py','--data',str(a.data),'--out',str(a.out/(name+'_'+phase)),
                      '--phase',phase,'--model',model,'--max-seconds','900']
                if phase=='select':argv+=['--frozen',str(a.out/(name+'_train')/'summary.json')]
                stage(name+'_'+phase,argv,1000)
        ready={'engineering_checks_passed':True,'train_select_only':True,'cloud_used':False,'test_used':False,
              'main_paper_ready':False,'ready_for_large_cloud_acquisition':False,
              'pending':['Live resource/clock/capture-lifecycle preflight',
                         'Privacy attacker training and frozen selection on matching observer data',
                         'Per-site observer design versus multiplexed tunnel study',
                         'Physical joule measurement if manuscript energy claims retained'],
              'stages':stages}
        atomic_json(a.out/'readiness.json',ready)
        sources={p.name:p for p in a.out.iterdir() if p.name not in ['backups','backup_audit.json']}
        backup=snapshot(sources,a.out.parent/'stage2_backups',a.out.name)
        drill=a.out.parent/(a.out.name+'_restore_drill')
        restored=verify(a.out.parent/'stage2_backups',a.out.name,drill)
        atomic_json(a.out/'backup_audit.json',{'snapshot':backup,'restore':restored,'temporary_restore_path':str(drill)})
        if not backup['complete'] or not restored['restore_tested']:raise RuntimeError('backup/restore verification failed')
        # This is only the newly created restore-test copy, never primary evidence.
        shutil.rmtree(drill)
        check_storage_budget(a.out,max_bytes=512*1024**2,reserve_bytes=2*1024**3,upcoming_bytes=0)
        atomic_json(a.out/'status.json',{'stage':'completed','cloud_used':False,'test_used':False,'main_paper_ready':False})
    except BaseException as exc:
        atomic_json(a.out/'failure.json',{'error':repr(exc),'cloud_used':False,'test_used':False,'automatic_retry':False})
        atomic_json(a.out/'readiness.json',{'engineering_checks_passed':False,'main_paper_ready':False,'cloud_used':False,'test_used':False})
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        atomic_json(a.out/'source_sha256.json',{str(p.relative_to(app)):file_sha256(p)
                    for folder in ['src','scripts'] for p in (app/folder).rglob('*.py')})


if __name__=='__main__':main()
