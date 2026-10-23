"""Two-model preparation uses this bounded TRAIN/select structured-answer check.

Actual accepted typed plans run over minimal TLS; three training steps are an
explicit controller workflow extension. It is not free-text factual QA, hosted
model validation, LangGraph validation, or a privacy evaluation.
"""
import argparse
from dataclasses import replace
from decimal import Decimal
import json
import math
import shutil
import time
import urllib.request
from pathlib import Path

from validate_higgs_planner import source_contract as planner_contract, guard_partition
from mac_stage2_calibration import contract as workflow_contract, input_contract
from greenshaping.testbed.higgs_planner import planner_prompt, question_schema, validate_plan
from greenshaping.testbed.higgs_workload import site_table
from greenshaping.testbed.higgs_gateway import RoutedTool, routed_calls
from greenshaping.testbed.minimal_tls import MinimalClient, MinimalServer
from greenshaping.testbed.question_bank import load_question_bank
from greenshaping.testbed.queries import run_query
from greenshaping.testbed.verified_workflow import execute
from greenshaping.utils.evidence import SessionLedger, atomic_json
from greenshaping.utils.storage_budget import check as check_storage_budget
from greenshaping.utils.provenance import file_sha256


def answer_schema(value):
    if isinstance(value,dict):return {'type':'object','properties':{k:answer_schema(v) for k,v in value.items()},'required':list(value),'additionalProperties':False}
    if isinstance(value,list):return {'type':'array','items':{'type':'number'},'minItems':len(value),'maxItems':len(value)}
    return {'type':'integer' if type(value) is int else 'number'}


def decimal_json(value):
    """Serialize verified floats as decimal JSON tokens without exponent notation.

    This changes presentation only; the reference values and matcher stay intact.
    """
    if isinstance(value,dict):
        return '{'+','.join(json.dumps(k)+':'+decimal_json(v) for k,v in value.items())+'}'
    if isinstance(value,list):return '['+','.join(decimal_json(v) for v in value)+']'
    if type(value) is float:
        if not math.isfinite(value):raise ValueError('nonfinite verified result')
        return format(Decimal(repr(value)),'f')
    return json.dumps(value,allow_nan=False)


def answer_matches(actual,expected):
    if isinstance(expected,dict):return isinstance(actual,dict) and actual.keys()==expected.keys() and all(answer_matches(actual[k],v) for k,v in expected.items())
    if isinstance(expected,list):return isinstance(actual,list) and len(actual)==len(expected) and all(answer_matches(a,b) for a,b in zip(actual,expected))
    if type(expected) is int:return type(actual) is int and actual==expected
    return type(actual) in (int,float) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-6,abs_tol=1e-8)


def chat(model,prompt,schema):
    request={'model':model,'stream':False,'format':schema,'messages':[{'role':'user','content':prompt}],
             'options':{'temperature':0,'seed':20261021,'num_predict':2048,'num_ctx':8192}}
    req=urllib.request.Request('http://127.0.0.1:11434/api/chat',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as r:response=json.load(r)
    return request,response


def main():
    import subprocess
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--phase',choices=['train','select'],default='train')
    p.add_argument('--frozen',type=Path);p.add_argument('--max-seconds',type=int,default=1800)
    a=p.parse_args()
    if not 30<=a.max_seconds<=2400:p.error('bounded 30..2400 seconds required')
    if shutil.disk_usage(a.out.parent if a.out.parent.exists() else '.').free<4*1024**3:raise RuntimeError('>=4 GiB free required for bounded structured-answer validation')
    questions=load_question_bank(a.data/'banks/n20'/(a.phase+'.jsonl'),require_min=False)
    frozen=json.loads(a.frozen.read_text()) if a.frozen else None
    guard_partition(questions,a.phase,frozen)
    local_contract=planner_contract()|{'answer_runner_sha256':file_sha256(Path(__file__)), 'workflow_contract':workflow_contract()}
    if a.phase=='select' and (frozen.get('answer_contract')!=local_contract or frozen.get('model')!=a.model or frozen.get('input_contract')!=input_contract(a.data,20)):raise ValueError('answer source/model changed')
    chosen={}
    for q in questions:chosen.setdefault((q.family,q.metadata['requested_size']),q)
    if len(chosen)!=12:raise ValueError('need 12 strata')
    a.out.mkdir(parents=True,exist_ok=False)
    ledger=SessionLedger(a.out/'ledger',{'model':a.model,'partition':a.phase,'contract':local_contract,'purpose':'structured summary consistency','no_fallback':True,'test_used':False})
    cert,key=a.out/'tls.cert',a.out/'tls.key'
    subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1','-subj','/CN=127.0.0.1','-addext','subjectAltName=IP:127.0.0.1','-keyout',str(key),'-out',str(cert)],capture_output=True,check=True);key.chmod(0o600)
    gateway=RoutedTool(set(range(20)),lambda i:site_table(a.data/'store',20,i))
    for i in range(20):gateway.tables[i]=site_table(a.data/'store',20,i)
    server=MinimalServer(str(cert),str(key),gateway);deadline=time.monotonic()+a.max_seconds;records=[]
    try:
        for q in chosen.values():
            if time.monotonic()>deadline:raise TimeoutError('no further model dispatch after deadline')
            check_storage_budget(a.out)
            row={'question_id':q.question_id,'model':a.model,'start_epoch':time.time(),'passed':False};sid=str(q.question_id);ledger.start(sid,row)
            try:
                prompt=planner_prompt(q.text,20)+'\nReturn the initial calls only. For train_round the declared controller then performs 3 weighted federated updates with the same sites; initial weights remain zero.'
                request,response=chat(a.model,prompt,question_schema(q.text,20));row.update(planner_request=request,planner_response=response,planning_done_epoch=time.time())
                plan=validate_plan(json.loads(response['message']['content']),20,q.text)
                if plan!={'family':q.family,'calls':q.plan[0]}:raise ValueError('accepted model plan differs from question reference; no fallback')
                actual_q=replace(q, plan=[plan['calls']])
                client=MinimalClient([server.address]*20,str(cert))
                try:actual=execute(actual_q,lambda f,c:client.call_round(f,routed_calls(c)))
                finally:client.close()
                row.update(tool_completed_epoch=time.time(),actual_workflow=actual)
                def direct(f,calls):return [dict(zip(('result','matching_rows'),run_query(gateway.tables[c['site']],f,c['params']))) for c in calls]
                reference=execute(q,direct)
                if not answer_matches(actual['answer'],reference['answer']):raise ValueError('tool workflow reference mismatch')
                request,response=chat(a.model,'Report the verified tool result as JSON without rounding or adding fields. Copy each number exactly, including its sign and leading fractional zeros. Use plain decimal notation, without e/E exponent notation. This is the final structured result of the analyst task.\nAnalyst: '+q.text+'\nController rounds: '+str(len(actual['rounds']))+'\nVerified tool result: '+decimal_json(actual['answer']),answer_schema(actual['answer']))
                answer=json.loads(response['message']['content']);row.update(answer_request=request,answer_response=response,answer=answer,
                    final_answer_epoch=time.time(),passed=answer_matches(answer,reference['answer']))
            except Exception as exc:row['error']=repr(exc)
            row['end_epoch']=time.time();ledger.commit(sid,row);records.append(row)
            atomic_json(a.out/'progress.json',{'completed':len(records),'passed':sum(r['passed'] for r in records),'test_used':False})
        summary={'model':a.model,'partition':a.phase,'questions':12,'passed':sum(r['passed'] for r in records),
            'source_contract':planner_contract(),'answer_contract':local_contract,'input_contract':input_contract(a.data,20),'test_used':False,
            'scope':'Typed first plan + deterministic three-step training workflow + structured numerical summary; not free-text QA or autonomous replanning',
            'audit':ledger.checkpoint()}
        atomic_json(a.out/'summary.json',summary)
        if summary['passed']!=12:raise SystemExit(2)
    finally:server.close()


if __name__=='__main__':main()
