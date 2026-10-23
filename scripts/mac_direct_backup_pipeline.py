"""Verify a direct Mac-to-Drive backup, then perform guarded source cleanup."""
import hashlib,json,os,pathlib,shutil,subprocess,sys,time,traceback,fcntl,configparser,urllib.request,urllib.parse,urllib.error
from mac_full_backup_pipeline import BASE,OUT,PARENT,REMOTE,RCLONE,PY,RUNNER,STAGING,status,remote,call,drive

def api_inventory(folder):
 c=configparser.ConfigParser();c.read(pathlib.Path.home()/'.config/rclone/rclone.conf');token=json.loads(c['gdrive']['token'])['access_token']
 query=urllib.parse.urlencode({'q':"'%s' in parents and trashed=false"%folder,'fields':'files(id,name,size,md5Checksum),nextPageToken','pageSize':1000})
 req=urllib.request.Request('https://www.googleapis.com/drive/v3/files?'+query,headers={'Authorization':'Bearer '+token})
 for attempt in range(30):
  try:
   with urllib.request.urlopen(req,timeout=90) as r:d=json.load(r)
   break
  except urllib.error.HTTPError as e:
   if e.code not in (403,429,500,502,503,504) or attempt==29:raise RuntimeError('Drive inventory HTTP '+str(e.code)) from None
   time.sleep(30)
 if d.get('nextPageToken'):raise ValueError('unexpected destination pagination')
 return d['files']

def main():
 meta=json.loads((OUT/'payload/PARTS.json').read_text());folder=json.loads((OUT/'direct_folder.json').read_text())
 status('direct_mac_to_drive_upload',total_parts=len(meta['parts']))
 for i in range(180):
  state=json.loads(remote("from pathlib import Path;import json;p=Path("+repr(STAGING)+");e=p/'direct_upload_error.json';s=p/'direct_upload_status.json';print(json.dumps({'error':json.loads(e.read_text()) if e.exists() else None,'progress':json.loads(s.read_text()) if s.exists() else {'completed':[]}}))"))
  (OUT/'mac_direct_upload_status.json').write_text(json.dumps(state,indent=2)+'\n')
  if state['error']:raise RuntimeError('Mac upload failed: '+str(state['error']))
  done=len(state['progress']['completed']);status('direct_mac_to_drive_upload',completed_parts=done,total_parts=len(meta['parts']))
  if done==len(meta['parts']):break
  time.sleep(20)
 else:raise TimeoutError('direct upload deadline; source preserved')
 md=OUT/'metadata';md.mkdir(exist_ok=True)
 shutil.copy2(OUT/'payload/PARTS.json',md/'PARTS.json');shutil.copy2(BASE/'scripts/verify_mac_full_backup.py',md/'verify_mac_full_backup.py')
 (md/'README_RESTORE_vi.txt').write_text('GREENSHAPING — BẢN SAO ĐẦY ĐỦ MAC STUDIO\n\nToàn bộ project: 8.023.906.252 byte dữ liệu file, 61.490 mục; gồm mã, kết quả, dữ liệu và môi trường Python. 22 phần mã hóa AES-256-GCM chứa archive tar.gz và manifest SHA-256 từng file. Symlink được giữ nguyên; đích bên ngoài project không được sao chép. MacControl chứa các LaunchAgents đã dừng.\n\nKHÓA RIÊNG KHÔNG ĐƯỢC TẢI LÊN DRIVE. Chủ sở hữu giữ khóa ở máy điều khiển tại outputs/mac_full_backup_20261022/private/RESTORE_PRIVATE_KEY.pem. Mất khóa sẽ không giải mã được. Hãy giữ bản sao khóa riêng ở nơi an toàn.\n\nCần Python 3 và thư viện cryptography. Khôi phục vào thư mục mới chưa tồn tại:\npython3 verify_mac_full_backup.py --parts . --private-key /path/to/RESTORE_PRIVATE_KEY.pem --restore-to /path/to/new_destination\n\nScript kiểm tra từng phần, xác thực AES-GCM, kiểm tra gzip và đối chiếu toàn bộ nội dung với manifest. Không tự khởi động lại LaunchAgents.\n\nBản thảo và kết quả EC2 riêng: https://drive.google.com/drive/folders/1Y6o5zNNVg4KE7Bt9ljVUt0_bV3UOh5pn\n',encoding='utf-8')
 expected={r['name']:r['sha256'] for r in meta['parts']}
 expected.update({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in md.iterdir() if p.name!='SHA256SUMS.txt'})
 (md/'SHA256SUMS.txt').write_text(''.join(h+'  '+n+'\n' for n,h in sorted(expected.items())))
 expected['SHA256SUMS.txt']=hashlib.sha256((md/'SHA256SUMS.txt').read_bytes()).hexdigest()
 status('uploading_restore_metadata')
 drive(['copy',str(md),REMOTE,'--checksum','--immutable','--transfers','2'],'direct_metadata_upload.log')
 # Metadata is checked separately; complete encrypted parts are then downloaded and SHA-verified.
 inv=api_inventory(folder['id']);(OUT/'direct_drive_inventory.json').write_text(json.dumps(inv,indent=2)+'\n')
 if len(inv)!=len(expected) or {r['name'] for r in inv}!=set(expected):raise ValueError('Drive inventory mismatch')
 indexed={r['name']:r for r in inv}
 for row in meta['parts']:
  r=indexed[row['name']]
  if int(r['size'])!=row['bytes'] or r['md5Checksum']!=row['md5']:raise ValueError('Drive size/MD5 mismatch')
 readback=OUT/'drive_readback_payload';readback.mkdir(exist_ok=True)
 status('downloading_drive_for_full_readback',files=len(expected),bytes=sum(int(r['size']) for r in inv))
 drive(['copy',REMOTE,str(readback),'--checksum','--transfers','3','--checkers','3','--stats','30s'],'direct_drive_readback.log')
 for name,wanted in expected.items():
  h=hashlib.sha256()
  with (readback/name).open('rb') as f:
   for b in iter(lambda:f.read(1024**2),b''):h.update(b)
  if h.hexdigest()!=wanted:raise ValueError('Drive readback SHA-256 mismatch: '+name)
 status('verifying_restore_from_drive_readback')
 result=json.loads(call([PY,str(BASE/'scripts/verify_mac_full_backup.py'),'--parts',str(readback),'--private-key',str(OUT/'private/RESTORE_PRIVATE_KEY.pem')],timeout=1800))
 if not result['verified'] or not result['authenticated']:raise ValueError('archive restore verification failed')
 (OUT/'archive_verification.json').write_text(json.dumps(result,indent=2)+'\n')
 proof={'drive_readback_verified':True,'archive_restore_verified':True,'manifest_sha256':meta['manifest_sha256'],'drive_folder_id':folder['id'],'drive_url':folder['url'],'uploaded_files':len(expected),'uploaded_bytes':sum(int(r['size']) for r in inv),'archive_verification':result,'verified_epoch':time.time(),'readback_method':'All Drive files downloaded; SHA-256 matched source; complete authenticated archive decoded and all source contents verified'}
 (OUT/'DRIVE_BACKUP_VERIFIED.json').write_text(json.dumps(proof,indent=2)+'\n')
 status('verified_preparing_guarded_mac_cleanup',proof=proof)
 call(RUNNER+['put',str(BASE/'scripts/mac_backup_cleanup.py'),STAGING+'/cleanup.py'])
 receipt=json.loads(call(RUNNER+['exec','/usr/bin/python3 '+STAGING+'/cleanup.py','--timeout','600'],timeout=630,input=json.dumps(proof)))
 (OUT/'MAC_DELETION_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n')
 check=json.loads(remote("import pathlib,json,shutil;print(json.dumps({'project_exists':pathlib.Path('/Users/sonha/Projects/GreenShaping').exists(),'staging_exists':pathlib.Path("+repr(STAGING)+").exists(),'free_bytes':shutil.disk_usage('/Users/sonha/Projects').free}))"))
 if check['project_exists'] or check['staging_exists']:raise ValueError('Mac deletion postcheck failed')
 # Remove only incomplete ciphertext from the superseded slow transfer.
 for p in (OUT/'payload').glob('*.partial'):p.unlink()
 status('complete',deletion_performed=True,receipt=receipt,postcheck=check)

if __name__=='__main__':
 try:
  lock=(OUT/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (OUT/'MAC_DELETION_VERIFIED.json').exists():sys.exit(0)
  main()
 except Exception as e:
  status('stopped_requires_attention',error=repr(e));traceback.print_exc();sys.exit(1)
