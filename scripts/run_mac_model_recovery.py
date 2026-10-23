"""Model-only recovery: immutable calibration inputs, finite independent stages."""
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
from greenshaping.utils.evidence_backup import snapshot, verify
from greenshaping.utils.provenance import file_sha256
from greenshaping.utils.storage_budget import check


def inventory(root):
    return {str(f.relative_to(root)):file_sha256(f) for f in root.rglob('*') if f.is_file()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    r=a.root;out=r/'results';out.mkdir(exist_ok=False);app=r/'app'
    old=Path('/Users/sonha/Projects/GreenShaping/stage2_preparation_20261021_09/results')
    data=Path('/Users/sonha/Projects/GreenShaping/large_preparation_20261021/evidence')
    protected={name:old/name for name in ['train','select','engineering_selection.json']}
    before={name:inventory(path) if path.is_dir() else file_sha256(path) for name,path in protected.items()}
    atomic_json(out/'calibration_before.json',before)
    stages={};child=None;deadline=time.time()+7200
    def stop(*_):raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def stage(name,argv,limit):
        nonlocal child
        check(out,max_bytes=256*1024**2,reserve_bytes=4*1024**3,upcoming_bytes=32*1024**2)
        if time.time()>=deadline:raise TimeoutError('recovery deadline')
        atomic_json(out/'status.json',{'stage':name,'deadline_epoch':deadline,'cloud_used':False,'test_used':False})
        with (out/(name+'.log')).open('wb') as log:
            child=subprocess.Popen(argv,cwd=app,stdout=log,stderr=log,start_new_session=True,
                env=dict(os.environ,PYTHONPATH=str(app/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            try:code=child.wait(timeout=min(limit,max(1,deadline-time.time())))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=124
        child=None;stages[name]={'returncode':code};atomic_json(out/'stages.json',stages)
        snap=snapshot({name:out/name} if (out/name).is_dir() else {name+'.log':out/(name+'.log')},r/'backups','stage_'+name)
        if not snap['complete']:raise RuntimeError('incomplete evidence snapshot')
        return code
    try:
        rc=stage('regression',[sys.executable,'-m','pytest','-q','tests/test_answer_workflow.py','tests/test_evidence_durability.py','tests/test_evidence_replica.py'],300)
        if rc:raise RuntimeError('focused regression failed')
        manifest=json.loads((data/'store/projection.json').read_text())
        if {f:file_sha256(data/'store'/f) for f in manifest['files']}!=manifest['files']:raise ValueError('projected data integrity')
        for model,name in [('mistral-small3.1:24b','mistral'),('qwen2.5:14b-instruct-q4_K_M','qwen14b')]:
            train_ok=False
            for phase in ['train','select']:
                if phase=='select' and not train_ok:
                    stages[name+'_select']={'skipped':'TRAIN failed; do not dispatch SELECT'};atomic_json(out/'stages.json',stages);continue
                argv=[sys.executable,'scripts/validate_higgs_answer_workflow.py','--data',str(data),'--out',str(out/(name+'_'+phase)),'--model',model,'--phase',phase,'--max-seconds','1200']
                if phase=='select':argv+=['--frozen',str(out/(name+'_train/summary.json'))]
                rc=stage(name+'_'+phase,argv,1400)
                if phase=='train':train_ok=rc==0
    except BaseException as e:
        atomic_json(out/'failure.json',{'error':repr(e),'automatic_retry':False})
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        after={name:inventory(path) if path.is_dir() else file_sha256(path) for name,path in protected.items()}
        unchanged=before==after
        atomic_json(out/'calibration_unchanged.json',{'unchanged':unchanged,'after':after})
        atomic_json(out/'source_sha256.json',{str(f.relative_to(app)):file_sha256(f) for folder in ['src','scripts'] for f in (app/folder).rglob('*.py')})
        try:
            sources={'recovery':out}|{'calibration_'+name:path for name,path in protected.items()}
            snap=snapshot(sources,r/'backups','final')
            drill=r/'restore_drill';audit=verify(r/'backups','final',drill)
            # Independently hash restored bytes, not only backup objects.
            mf=json.loads((r/'backups/final.json').read_text())
            restored=all(file_sha256(drill/name)==spec['sha256'] for name,spec in mf['files'].items())
            atomic_json(out/'backup_audit.json',{'snapshot':snap,'verification':audit,'restored_bytes_verified':restored})
            if restored and audit['complete']:shutil.rmtree(drill)
            backup_ok=restored and audit['complete']
        except BaseException as e:
            backup_ok=False;atomic_json(out/'backup_failure.json',{'error':repr(e)})
        expected=['regression','mistral_train','mistral_select','qwen14b_train','qwen14b_select']
        passed=all(stages.get(n,{}).get('returncode')==0 for n in expected) and unchanged and backup_ok and not (out/'failure.json').exists()
        result={'stage':'completed' if passed else 'completed_with_findings','model_checks_passed':passed,'calibration_unchanged':unchanged,'restore_verified':backup_ok,'cloud_used':False,'test_used':False,'main_paper_ready':False,'ready_for_large_cloud_acquisition':False,'stages':stages}
        atomic_json(out/'readiness.json',result);atomic_json(out/'status.json',result)
        (out/'MORNING_REPORT_vi.txt').write_text('Kiểm tra mô hình: '+('ĐẠT' if passed else 'CÒN VẤN ĐỀ')+'\nHiệu chuẩn giữ nguyên: '+str(unchanged)+'\nPhục hồi xác minh: '+str(backup_ok)+'\nXem readiness.json, stages.json và từng summary/ledger. Không tự bật EC2.\n')
    return 0 if passed else 2


if __name__=='__main__':sys.exit(main())
