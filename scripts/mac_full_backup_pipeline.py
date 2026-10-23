"""Bounded resumable backup; deletion gated on archive and Drive readback checks."""
import hashlib,json,os,pathlib,shlex,shutil,subprocess,sys,time,traceback
BASE=pathlib.Path(__file__).resolve().parents[1]
OUT=BASE/'outputs/mac_full_backup_20261022'
PAYLOAD=OUT/'payload'
STAGING='/Users/sonha/Projects/GreenShaping_full_backup_20261022'
PARENT='1nQ4JSto8DGKQlVlW8sW_J3uyKgP9hWnJ'
CHILD='MacStudio_GreenShaping_full_20261022'
REMOTE='gdrive:'+CHILD
RCLONE=str(pathlib.Path.home()/'.local/bin/rclone')
PY=str(BASE/'.venv/bin/python')
RUNNER=[PY,str(BASE/'scripts/remote_runner.py'),'--known-hosts',str(BASE/'outputs/remote_20261020/known_hosts')]
DEADLINE=time.monotonic()+6*3600

def status(stage,**kw):
 d=dict(stage=stage,updated_epoch=time.time(),deletion_performed=False);d.update(kw)
 tmp=OUT/'status.tmp';tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(OUT/'status.json');print(stage,flush=True)

def call(cmd,timeout=300,input=None):
 p=subprocess.run(cmd,input=input,text=True,capture_output=True,timeout=timeout,cwd=BASE)
 if p.returncode:raise RuntimeError(str(cmd[:2])+' failed: '+p.stderr[-1800:]+p.stdout[-1000:])
 return p.stdout

def remote(code,timeout=60):return call(RUNNER+['exec','/usr/bin/python3 -','--timeout',str(timeout)],timeout=timeout+30,input=code)

def get(source,target,sha=None):
 if target.exists() and sha:
  h=hashlib.sha256()
  with target.open('rb') as f:
   for b in iter(lambda:f.read(1024**2),b''):h.update(b)
  if h.hexdigest()==sha:return
 args=RUNNER+['get',source,str(target)]
 if sha:args+=['--sha256',sha]
 call(args,timeout=1800)

def drive(args,logname,attempts=12):
 for attempt in range(attempts):
  if time.monotonic()>DEADLINE:raise TimeoutError('backup deadline; source preserved')
  with (OUT/logname).open('a') as log:
   log.write('\nattempt '+str(attempt+1)+'\n');log.flush()
   p=subprocess.run([RCLONE]+args+['--drive-root-folder-id',PARENT,'--tpslimit','1','--timeout','60s','--retries','2','--low-level-retries','3','--retries-sleep','10s'],stdout=log,stderr=subprocess.STDOUT,timeout=1800)
  if p.returncode==0:return
  status('drive_retry',operation=args[0],attempt=attempt+1)
  time.sleep(30)
 raise RuntimeError('Drive operation failed; source preserved: '+args[0])

def main():
 PAYLOAD.mkdir(exist_ok=True)
 status('waiting_for_mac_snapshot')
 for i in range(90):
  text=remote("from pathlib import Path; p=Path("+repr(STAGING+'/snapshot02/status.json')+"); print(p.read_text() if p.exists() else '{}')")
  state=json.loads(text)
  (OUT/'mac_snapshot_status.json').write_text(json.dumps(state,indent=2)+'\n')
  if state.get('stage')=='snapshot_complete':break
  if state.get('stage')=='snapshot_failed':raise RuntimeError(str(state))
  time.sleep(20)
 else:raise TimeoutError('snapshot deadline; source preserved')
 status('copying_encrypted_snapshot_to_controller',snapshot=state)
 get(STAGING+'/snapshot02/payload/PARTS.json',PAYLOAD/'PARTS.json')
 meta=json.loads((PAYLOAD/'PARTS.json').read_text())
 get(STAGING+'/snapshot02/manifest.json',OUT/'manifest.json',meta['manifest_sha256'])
 if shutil.disk_usage(OUT).free<sum(p['bytes'] for p in meta['parts'])+5*1024**3:raise RuntimeError('insufficient controller disk; source preserved')
 for i,row in enumerate(meta['parts']):
  status('copying_encrypted_snapshot_to_controller',part=i+1,total_parts=len(meta['parts']))
  get(STAGING+'/snapshot02/payload/'+row['name'],PAYLOAD/row['name'],row['sha256'])
 status('verifying_complete_archive')
 result=json.loads(call([PY,str(BASE/'scripts/verify_mac_full_backup.py'),'--parts',str(PAYLOAD),'--private-key',str(OUT/'private/RESTORE_PRIVATE_KEY.pem')],timeout=1800))
 (OUT/'archive_verification.json').write_text(json.dumps(result,indent=2)+'\n')
 if not result['verified'] or not result['authenticated']:raise RuntimeError('archive failed verification')
 shutil.copy2(BASE/'scripts/verify_mac_full_backup.py',PAYLOAD/'verify_mac_full_backup.py')
 (PAYLOAD/'README_RESTORE_vi.txt').write_text('BẢN SAO ĐẦY ĐỦ GREENSHAPING TRÊN MAC STUDIO\n\nToàn bộ nội dung project (mã, dữ liệu, kết quả, môi trường Python, symlink) nằm trong các phần mã hóa AES-256-GCM. Không bỏ bất kỳ phần nào. PARTS.json lưu thứ tự và SHA-256.\n\nKhóa khôi phục RSA RIÊNG KHÔNG có trên Drive. Chủ sở hữu giữ khóa tại máy điều khiển: outputs/mac_full_backup_20261022/private/RESTORE_PRIVATE_KEY.pem. Cần sao lưu khóa riêng ở nơi an toàn. Mất khóa này sẽ không giải mã được. Không chia sẻ công khai khóa riêng.\n\nKhôi phục bằng Python 3 và thư viện cryptography:\npython3 verify_mac_full_backup.py --parts . --private-key /path/to/RESTORE_PRIVATE_KEY.pem --restore-to /path/to/new_empty_destination\n\nScript kiểm tra SHA-256 từng phần, xác thực AES-GCM và đối chiếu tất cả file với manifest trong archive. Symlink được giữ nguyên; các đích bên ngoài project và mô hình dùng chung không nằm trong gói này. Thư mục MacControl chứa LaunchAgents đã vô hiệu hóa; chỉ khôi phục tác vụ nếu thực sự cần.\n\nKết quả EC2 và bản thảo đã được sao lưu riêng tại https://drive.google.com/drive/folders/1Y6o5zNNVg4KE7Bt9ljVUt0_bV3UOh5pn .\n',encoding='utf-8')
 rows=[]
 for p in sorted(PAYLOAD.iterdir()):
  if p.name=='SHA256SUMS.txt':continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024**2),b''):h.update(b)
  rows.append(h.hexdigest()+'  '+p.name)
 (PAYLOAD/'SHA256SUMS.txt').write_text('\n'.join(rows)+'\n')
 status('uploading_complete_archive_to_drive',bytes=sum(p.stat().st_size for p in PAYLOAD.iterdir()))
 drive(['mkdir',REMOTE],'drive_mkdir.log')
 drive(['copy',str(PAYLOAD),REMOTE,'--checksum','--immutable','--transfers','2','--checkers','2','--stats','30s'],'drive_upload.log')
 status('verifying_drive_readback')
 drive(['check',str(PAYLOAD),REMOTE,'--download','--one-way','--checkers','2','--combined',str(OUT/'drive_readback.txt')],'drive_readback.log')
 report=(OUT/'drive_readback.txt').read_text().splitlines();expected={p.name for p in PAYLOAD.iterdir() if p.is_file()}
 matched={line[2:] for line in report if line.startswith('= ')}
 if matched!=expected or any(not line.startswith('= ') for line in report):raise RuntimeError('Drive readback inventory mismatch; source preserved')
 # Read folder metadata without changing its sharing settings.
 drive(['lsjson',REMOTE,'--stat'],'drive_folder_metadata.log')
 folder=json.loads((OUT/'drive_folder_metadata.log').read_text().rsplit('\nattempt ',1)[1].split('\n',1)[1])
 proof={'drive_readback_verified':True,'archive_restore_verified':True,'manifest_sha256':meta['manifest_sha256'],'drive_folder_id':folder['ID'],'drive_url':'https://drive.google.com/drive/folders/'+folder['ID'],'uploaded_files':len(expected),'uploaded_bytes':sum(p.stat().st_size for p in PAYLOAD.iterdir()),'archive_verification':result,'verified_epoch':time.time()}
 (OUT/'DRIVE_BACKUP_VERIFIED.json').write_text(json.dumps(proof,indent=2)+'\n')
 status('verified_preparing_guarded_mac_cleanup',proof=proof)
 call(RUNNER+['put',str(BASE/'scripts/mac_backup_cleanup.py'),STAGING+'/cleanup.py'])
 receipt=json.loads(call(RUNNER+['exec','/usr/bin/python3 '+shlex.quote(STAGING+'/cleanup.py'),'--timeout','600'],timeout=630,input=json.dumps(proof)))
 (OUT/'MAC_DELETION_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n')
 check=json.loads(remote("import pathlib,json,shutil;print(json.dumps({'project_exists':pathlib.Path('/Users/sonha/Projects/GreenShaping').exists(),'staging_exists':pathlib.Path("+repr(STAGING)+").exists(),'free_bytes':shutil.disk_usage('/Users/sonha/Projects').free}))"))
 if check['project_exists'] or check['staging_exists']:raise RuntimeError('cleanup postcheck failed')
 status('complete',deletion_performed=True,receipt=receipt,postcheck=check)

if __name__=='__main__':
 try:
  import fcntl
  lock=(OUT/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (OUT/'MAC_DELETION_VERIFIED.json').exists():sys.exit(0)
  main()
 except Exception as e:
  status('stopped_requires_attention',error=repr(e));traceback.print_exc();sys.exit(1)
