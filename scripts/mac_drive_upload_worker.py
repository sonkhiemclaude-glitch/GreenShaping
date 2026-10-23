"""Bounded resumable upload using file-specific session URLs; never logs URLs."""
import concurrent.futures,json,pathlib,time,urllib.request,urllib.error,random
ROOT=pathlib.Path('/Users/sonha/Projects/GreenShaping_full_backup_20261022')
DEADLINE=time.monotonic()+3*3600
CHUNK_BYTES=64*1024**2

def attempt(req):
 try:
  with urllib.request.urlopen(req,timeout=180) as r:return ('complete',json.loads(r.read()))
 except urllib.error.HTTPError as e:
  if e.code==308:
   r=e.headers.get('Range','');return ('offset',int(r.split('-')[-1])+1 if r else 0)
  transient=e.code in (429,500,502,503,504)
  if e.code==403:
   try:transient=any(x.get('reason') in ('rateLimitExceeded','userRateLimitExceeded') for x in json.loads(e.read()).get('error',{}).get('errors',[]))
   except Exception:pass
  if transient:return ('retry',e.code)
  raise RuntimeError('upload HTTP '+str(e.code)) from None
 except (OSError,TimeoutError):return ('retry','transport')

def upload(row):
 source=ROOT/'snapshot02/payload'/row['name'];size=source.stat().st_size
 if size!=row['bytes']:raise ValueError('source size changed')
 def query():
  while time.monotonic()<DEADLINE:
   q=urllib.request.Request(row['url'],data=b'',method='PUT',headers={'Content-Length':'0','Content-Range':'bytes */'+str(size)})
   result=attempt(q)
   if result[0]!='retry':return result
   time.sleep(30+random.random()*10)
  raise TimeoutError('upload deadline')
 state,value=query()
 if state=='complete':return {'name':row['name'],'id':value['id'],'bytes':size,'completed':True}
 offset=value
 with source.open('rb') as f:
  while offset<size and time.monotonic()<DEADLINE:
   f.seek(offset);data=f.read(min(CHUNK_BYTES,size-offset));end=offset+len(data)-1
   q=urllib.request.Request(row['url'],data=data,method='PUT',headers={'Content-Length':str(len(data)),'Content-Range':'bytes %d-%d/%d'%(offset,end,size),'Content-Type':'application/octet-stream'})
   state,value=attempt(q)
   if state=='retry':
    time.sleep(30+random.random()*10);state,value=query()
   if state=='complete':return {'name':row['name'],'id':value['id'],'bytes':size,'completed':True}
   if not 0<=value<=size:raise ValueError('unexpected server offset')
   offset=value
 raise TimeoutError('upload deadline or missing final metadata')

def main():
 rows=json.loads((ROOT/'upload_sessions.json').read_text());done=[]
 error=ROOT/'direct_upload_error.json'
 if error.exists():error.unlink()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  pending={pool.submit(upload,r):r for r in rows}
  for future in concurrent.futures.as_completed(pending):
   try:done.append(future.result())
   except Exception as e:
    error.write_text(json.dumps({'name':pending[future]['name'],'error_type':type(e).__name__,'reason':str(e) if isinstance(e,(RuntimeError,TimeoutError)) else 'transfer error'}))
    raise RuntimeError('upload failed: '+pending[future]['name']) from None
   temp=ROOT/'direct_upload_status.tmp';temp.write_text(json.dumps({'completed':done,'total':len(rows)}));temp.replace(ROOT/'direct_upload_status.json')
 print('all_encrypted_parts_uploaded',len(done),flush=True)

if __name__=='__main__':main()
