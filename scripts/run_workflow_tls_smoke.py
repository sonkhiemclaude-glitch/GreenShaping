"""Bounded local three-endpoint TLS smoke; all keys ephemeral, no captures/J."""
import json,subprocess,tempfile,time
from pathlib import Path
from greenshaping.controller.workflow_contract import WorkflowEnvelope
from greenshaping.testbed.workflow_cover import EnvelopeTLSSite,run_envelope_tls
OUT=Path('outputs/reliability_controller_20261023');cases=[]
with tempfile.TemporaryDirectory(prefix='gs-envelope-tls-') as temp:
 p=Path(temp);cert=p/'cert.pem';key=p/'key.pem'
 subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1','-subj','/CN=127.0.0.1','-addext','subjectAltName=IP:127.0.0.1','-keyout',str(key),'-out',str(cert)],check=True,capture_output=True)
 for name,actual_rounds,site_index,late in [('early_one_round',1,0,False),('three_round_feedback',3,0,False),('different_private_site',2,2,False),('over_bound_tool',1,0,True)]:
  e=WorkflowEnvelope(3,.1,.1,.1,(256,)*3,(512,)*3)
  def handler(request):
   time.sleep(.35 if late else .01);return request['x']*2
  sites=[EnvelopeTLSSite(e,i,cert,key,handler) for i in range(3)]
  def planner(r,previous):
   if r>=actual_rounds:return [None]*3
   if r==0:x=2
   else:x=previous[site_index]['result']
   values=[None]*3;values[site_index]={'x':x};return values
  result=run_envelope_tls(e,[s.address for s in sites],cert,planner)
  for s in sites:s.close()
  semantically_correct=all(result['responses'][r][site_index]=={'result':2**(r+2)} for r in range(actual_rounds)) if not late else result['responses'][0][site_index] is None
  frame_counts_ok=len(result['client_log'])==9 and sum(len(s.logs) for s in sites)==9
  size_ok=all(x['application_bytes']==256 for x in result['client_log']) and all(x['application_bytes']==512 for s in sites for x in s.logs)
  max_lag=max(x['lag_s'] for x in result['client_log']+[x for s in sites for x in s.logs]);site_errors=[x for s in sites for x in s.errors]
  passed=semantically_correct and frame_counts_ok and size_ok and max_lag<e.interval and (not result['completed'] and 'reply_not_ready' in site_errors if late else result['completed'] and not site_errors)
  row={'case':name,'expected_contract_failure':late,'semantic_reference_match':semantically_correct,'fixed_frame_counts_match':frame_counts_ok,'fixed_frame_sizes_match':size_ok,'max_send_completion_lag_s':max_lag,'passed':bool(passed),'client':{k:v for k,v in result.items() if k!='responses'},'site_logs':[s.logs for s in sites],'site_errors':site_errors};cases.append(row)
  (OUT/'local_tls_cases.json').write_text(json.dumps(cases,indent=2)+'\n');print(name,passed,flush=True)
summary={'scope':'three logical loopback TLS endpoints on controller host; fixed frames and semantic correctness only','cases':len(cases),'positive_cases_completed':sum(r['passed'] and not r['expected_contract_failure'] for r in cases),'negative_cases_rejected':sum(r['passed'] and r['expected_contract_failure'] for r in cases),'all_cases_passed':all(r['passed'] for r in cases),'max_send_completion_lag_s':max(r['max_send_completion_lag_s'] for r in cases),'wire_packets_captured':False,'physical_sites':1,'private_tls_keys_retained':False,'new_ec2_sessions':0,'final_bank_opened':False}
(OUT/'local_tls_summary.json').write_text(json.dumps(summary,indent=2)+'\n');assert summary['all_cases_passed'];print(json.dumps(summary))
