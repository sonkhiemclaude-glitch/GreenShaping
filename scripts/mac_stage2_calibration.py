"""Finite Mac-only TRAIN/select cost/utility calibration, never final test.

Produces a functional/cost shortlist, not a privacy-optimal or energy-optimal
selection. Invalid trials are retained and never retried as a new repetition.
"""
import argparse
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from mac_higgs_tls_check import equivalent
from greenshaping.shaping.pipeline import DPParams, PaddingParams, RoundsParams, ShapingConfig
from greenshaping.testbed.higgs_gateway import RoutedTool, routed_calls
from greenshaping.testbed.higgs_workload import site_table
from greenshaping.testbed.minimal_tls import MinimalServer, MinimalClient
from greenshaping.testbed.question_bank import load_question_bank
from greenshaping.testbed.queries import run_query
from greenshaping.testbed.transport import ShapedTransport, TLSToolServer
from greenshaping.testbed.verified_workflow import execute
from greenshaping.utils.evidence import SessionLedger, atomic_json
from greenshaping.utils.storage_budget import check as check_storage_budget
from greenshaping.utils.provenance import file_sha256


def configurations(n):
    configs={'minimal_tls':None,'none':ShapingConfig(), 'fixed_1024':ShapingConfig(fixed_unit=1024),
             'pad':ShapingConfig(padding=PaddingParams()),
             'rounds':ShapingConfig(rounds=RoundsParams(window=.5,cover_prob=1.,reply_frame=1024))}
    for name in ['dp','combo']:
        for mult in [0,1,2,4,8]:
            b=16384/(8/(2*n))
            dp=DPParams(eps_dir=8/(2*n),interval=.5,horizon=10.,mode='arrivals',delta=16384,
                        volume_cap_bytes=8192,bias=mult*b)
            configs[f'{name}_bias{mult}']=ShapingConfig(dp=dp,
                padding=PaddingParams() if name=='combo' else None,
                rounds=RoundsParams(window=.5,cover_prob=1.,reply_frame=1024) if name=='combo' else None)
    return configs


def contract():
    root=Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)):file_sha256(p) for p in [Path(__file__),
      root/'src/greenshaping/testbed/transport.py',root/'src/greenshaping/testbed/verified_workflow.py',
      root/'src/greenshaping/testbed/minimal_tls.py',root/'src/greenshaping/testbed/higgs_workload.py',
      root/'src/greenshaping/testbed/queries.py',root/'src/greenshaping/testbed/higgs_gateway.py']}


def input_contract(data,n):
    files=[data/'banks'/f'n{n}'/(p+'.jsonl') for p in ['train','select']]
    files.append(data/'store/projection.json')
    return {str(p.relative_to(data)):file_sha256(p) for p in files}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--N',type=int,choices=[5,10,20],default=20)
    p.add_argument('--phase',choices=['train','select'],default='train')
    p.add_argument('--frozen',type=Path);p.add_argument('--max-seconds',type=int,default=3000)
    p.add_argument('--repetitions',type=int,choices=[1,2],default=2)
    p.add_argument('--smoke-cases',type=int,default=0,help='TRAIN only: smaller diagnostic, never a shortlist')
    a=p.parse_args()
    if a.max_seconds>3600 or a.max_seconds<30:p.error('finite 30..3600 second window required')
    if shutil.disk_usage(a.out.parent if a.out.parent.exists() else '.').free<4*1024**3:
        raise RuntimeError('>=4 GiB free required for bounded JSON-only calibration; no pcap capture')
    bank=a.data/'banks'/f'n{a.N}'/(a.phase+'.jsonl')
    questions=load_question_bank(bank,require_min=False)
    if not questions or any(q.metadata.get('partition')!=a.phase for q in questions):raise ValueError('holdout guard')
    chosen={}
    for q in questions:chosen.setdefault((q.family,q.metadata['requested_size']),q)
    if len(chosen)!=12:raise ValueError('need 12 strata')
    configs=configurations(a.N);frozen=None
    if a.phase=='select':
        if a.smoke_cases or not a.frozen:p.error('select needs frozen TRAIN shortlist; no smoke')
        frozen=json.loads(a.frozen.read_text())
        if frozen['contract']!=contract() or frozen['N']!=a.N or frozen['test_used'] or frozen.get('input_contract')!=input_contract(a.data,a.N):
            raise ValueError('frozen source/N mismatch')
        names=frozen['shortlist']
        if any(name not in configs for name in names):raise ValueError('unknown frozen candidate')
    else:names=list(configs)
    if a.smoke_cases:
        if not 1<=a.smoke_cases<=12:p.error('invalid smoke case count')
        balanced=[];seen=set()
        for k,q in chosen.items():
            if q.family not in seen:balanced.append((k,q));seen.add(q.family)
        balanced.extend((k,q) for k,q in chosen.items() if k not in dict(balanced))
        chosen=dict(balanced[:a.smoke_cases]);names=['minimal_tls','none','fixed_1024','pad','rounds','dp_bias1','combo_bias1']
    a.out.mkdir(parents=True,exist_ok=False)
    ledger=SessionLedger(a.out/'ledger',{'contract':contract(),'N':a.N,'partition':a.phase,
        'bank_sha256':file_sha256(bank),'configurations':{k:v.to_dict() if v else None for k,v in configs.items()},
        'candidate_order':names,'repetitions':a.repetitions,'training_steps':3,'purpose':'functional/cost only','test_used':False})
    deadline=time.monotonic()+a.max_seconds;records=[];servers=[];client=None
    try:
        cert,key=a.out/'tls.cert',a.out/'tls.key'
        subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1','-subj','/CN=127.0.0.1',
                        '-addext','subjectAltName=IP:127.0.0.1','-keyout',str(key),'-out',str(cert)],capture_output=True,check=True)
        key.chmod(0o600)
        gateway=RoutedTool(set(range(a.N)),lambda i:site_table(a.data/'store',a.N,i))
        for i in range(a.N):gateway.tables[i]=site_table(a.data/'store',a.N,i)
        # Independent unshaped reference uses the same query definition but no network.
        def reference(f,calls):
            return [dict(zip(('result','matching_rows'),run_query(gateway.tables[c['site']],f,c['params']))) for c in calls]
        refs={q.question_id:execute(q,reference) for q in chosen.values()}
        server=TLSToolServer('127.0.0.1',0,str(cert),str(key),gateway);servers.append(server)
        minimal=MinimalServer(str(cert),str(key),gateway);servers.append(minimal)
        for name in names:
            cfg=configs[name]
            trials=[(rep,q) for rep in range(a.repetitions) for q in chosen.values()]
            for rep,q in trials:
                if time.monotonic()>deadline:raise TimeoutError('Mac-only acquisition deadline reached; no extension')
                check_storage_budget(a.out)
                sid=f'{name}_{rep}_{q.question_id}';row={'configuration':name,'question_id':q.question_id,'repetition':rep,
                    'partition':a.phase,'start_epoch':time.time(),'passed':False}
                ledger.start(sid,row);began=False
                try:
                    if client is None:
                        client=MinimalClient([minimal.address]*a.N,str(cert)) if cfg is None else ShapedTransport([server.address]*a.N,str(cert),cfg,start_lead_s=1.)
                    row['connection_ready_epoch']=time.time()
                    row['session_started_epoch']=time.time()
                    if cfg is not None:client.begin([c['site'] for c in q.plan[0]]);began=True
                    tx,rx=(client.tx,client.rx) if cfg is None else (0,0)
                    result=execute(q,lambda f,c:client.call_round(f,routed_calls(c)))
                    row.update(tool_completed_epoch=time.time(),answer=result['answer'],rounds=result['rounds'],
                               reference_match=equivalent([r['answer'] for r in result['rounds']],
                                                          [r['answer'] for r in refs[q.question_id]['rounds']]))
                    if cfg is None:
                        row['framed_plaintext_bytes']={'sent_data':client.tx-tx,'received_data':client.rx-rx,'sent_control':0,'received_control':0}
                        row['measurement_valid']=True
                except Exception as exc:row['error']=repr(exc)
                finally:
                    if began:
                        try:
                            row['transport']=client.finish();row['measurement_valid']=row['transport']['measurement_valid']
                            row['framed_plaintext_bytes']=row['transport']['framed_plaintext_bytes']
                        except Exception as exc:row['finish_error']=repr(exc)
                    row['end_epoch']=time.time()
                    row['passed']=row.get('measurement_valid',False) and row.get('reference_match',False) and 'error' not in row and 'finish_error' not in row
                    ledger.commit(sid,row);records.append(row)
                    atomic_json(a.out/'progress.json',{'completed':len(records),'passed':sum(r['passed'] for r in records),'last_id':sid,'test_used':False})
                if not row['passed']:
                    timing=row.get('transport',{})
                    channels=timing.get('upstream',[])+list(timing.get('downstream',{}).values())
                    if any(not c.get('timing_valid',True) for c in channels):
                        raise RuntimeError('Host scheduling/timing invalid; retained trial; halt instead of ranking configuration as infeasible')
                    if client:client.close();client=None
                    if not began and cfg is not None:raise RuntimeError('start barrier/client failed; halt acquisition')
            if client:client.close();client=None
        per={}
        for name in names:
            subset=[r for r in records if r['configuration']==name]
            per[name]={'trials':len(subset),'passed':sum(r['passed'] for r in subset),
                       'mean_framed_plaintext_bytes':sum(sum(r.get('framed_plaintext_bytes',{}).values()) for r in subset)/len(subset),
                       'all_passed':len(subset)==a.repetitions*len(chosen) and all(r['passed'] for r in subset)}
        shortlist=['minimal_tls','none','fixed_1024','pad','rounds']
        for prefix in ['dp_','combo_']:
            feasible=sorted((n for n in names if n.startswith(prefix) and per[n]['all_passed']),key=lambda n:per[n]['mean_framed_plaintext_bytes'])
            shortlist.extend(feasible[:2])
        if a.phase=='train' and not a.smoke_cases:
            atomic_json(a.out/'frozen_shortlist.json',{'contract':contract(),'input_contract':input_contract(a.data,a.N),'N':a.N,'shortlist':shortlist,
                'test_used':False,'purpose':'functional/cost candidates only; attacker and energy selection pending',
                'train_ledger_protocol_sha256':file_sha256(a.out/'ledger/protocol.json')})
        atomic_json(a.out/'summary.json',{'partition':a.phase,'N':a.N,'per_configuration':per,
             'all_baselines_passed':all(per[n]['all_passed'] for n in names if n in ['minimal_tls','none','fixed_1024','pad','rounds']),
             'test_used':False,'main_paper_ready':False,'privacy_selection_done':False,'energy_measured':False,'smoke_only':bool(a.smoke_cases)})
    finally:
        if client:client.close()
        for server in servers:
            if isinstance(server,MinimalServer):server.close()
            else:server.stop()
        atomic_json(a.out/'audit.json',ledger.checkpoint())


if __name__=='__main__':main()
