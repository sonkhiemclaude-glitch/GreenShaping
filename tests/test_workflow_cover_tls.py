import json,subprocess,time
from pathlib import Path
import pytest
from greenshaping.controller.workflow_contract import WorkflowEnvelope
from greenshaping.testbed.workflow_cover import EnvelopeTLSSite,run_envelope_tls,encode_fixed,decode_fixed

def make_certificate(root):
 cert=root/'cert.pem';key=root/'key.pem'
 subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1','-subj','/CN=127.0.0.1','-addext','subjectAltName=IP:127.0.0.1','-keyout',str(key),'-out',str(cert)],capture_output=True,check=True)
 return cert,key

def test_fixed_frames_reject_oversize_without_changing_size():
 assert decode_fixed(encode_fixed({'x':1},64))=={'x':1}
 assert decode_fixed(encode_fixed(None,64)) is None
 with pytest.raises(ValueError):encode_fixed({'x':'x'*100},64)

def test_three_tls_sites_keep_cover_after_private_early_completion(tmp_path):
 cert,key=make_certificate(tmp_path);e=WorkflowEnvelope(3,.08,.08,.08,(128,)*3,(256,)*3)
 def handler(request):return request['x']*2
 sites=[EnvelopeTLSSite(e,i,cert,key,handler) for i in range(3)]
 def planner(r,prior):
  if r==0:return [{'x':2},None,{'x':3}]
  if r==1:return [{'x':prior[0]['result']},None,None]
  return [None]*3
 result=run_envelope_tls(e,[s.address for s in sites],cert,planner)
 for s in sites:s.close()
 assert not result['errors'] and all(not s.errors for s in sites)
 assert result['responses'][0][0]['result']==4 and result['responses'][1][0]['result']==8
 assert all(v is None for v in result['responses'][2].values())
 assert len(result['client_log'])==9 and sum(len(s.logs) for s in sites)==9
 assert all(r['application_bytes']==128 for r in result['client_log'])
 assert all(r['application_bytes']==256 for s in sites for r in s.logs)
 assert max(r['lag_s'] for r in result['client_log']+[x for s in sites for x in s.logs])<e.interval

def test_slow_tool_keeps_fixed_reply_slots_and_is_not_counted_as_success(tmp_path):
 cert,key=make_certificate(tmp_path);e=WorkflowEnvelope(1,.08,.04,.08,(128,),(256,))
 def slow(request):time.sleep(.3);return 7
 site=EnvelopeTLSSite(e,0,cert,key,slow);result=run_envelope_tls(e,[site.address],cert,lambda r,p:[{'x':1}]);site.close()
 assert result['responses'][0][0] is None
 assert 'reply_not_ready' in site.errors
 assert len(site.logs)==1 and site.logs[0]['application_bytes']==256
