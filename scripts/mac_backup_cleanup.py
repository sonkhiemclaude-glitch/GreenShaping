"""Delete only an exact, unchanged project snapshot after external verification."""
import hashlib,json,os,pathlib,plistlib,shutil,stat,subprocess,time

ROOT=pathlib.Path('/Users/sonha/Projects/GreenShaping')
STAGING=pathlib.Path('/Users/sonha/Projects/GreenShaping_full_backup_20261022')

def unchanged(root,manifest):
 if root.is_symlink() or not root.is_dir():raise ValueError('root must be a real directory')
 actual={str(pathlib.Path(d,n).relative_to(root)) for d,ds,fs in os.walk(root,followlinks=False) for n in ds+fs}
 if actual!=set(manifest['entries']):raise ValueError('source inventory changed')
 for rel,e in manifest['entries'].items():
  p=root/rel;s=p.lstat()
  if stat.S_IMODE(s.st_mode)!=e['mode']:raise ValueError('source mode changed: '+rel)
  if e['type']=='file':
   if not stat.S_ISREG(s.st_mode) or s.st_size!=e['bytes'] or s.st_mtime_ns!=e['mtime_ns']:raise ValueError('source file changed: '+rel)
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(1024**2),b''):h.update(b)
   if h.hexdigest()!=e['sha256']:raise ValueError('source hash changed: '+rel)
  elif e['type']=='symlink':
   if not p.is_symlink() or os.readlink(p)!=e['target']:raise ValueError('source link changed: '+rel)
  elif not stat.S_ISDIR(s.st_mode):raise ValueError('source directory changed')
 return len(actual)

def cleanup(proof):
 if proof.get('drive_readback_verified') is not True or proof.get('archive_restore_verified') is not True:raise ValueError('verification required')
 manifest=json.loads((STAGING/'snapshot02/manifest.json').read_text())
 if manifest['root']!=str(ROOT):raise ValueError('wrong manifest root')
 if hashlib.sha256((STAGING/'snapshot02/manifest.json').read_bytes()).hexdigest()!=proof['manifest_sha256']:raise ValueError('wrong manifest checksum')
 ps=subprocess.check_output(['/bin/ps','-axo','pid=,command='],text=True)
 if any(str(ROOT)+'/' in row for row in ps.splitlines()):raise ValueError('project process is still running')
 saved=STAGING/'controls/LaunchAgents';original=pathlib.Path.home()/'Library/LaunchAgents';agents=[]
 for p in saved.glob('*.plist'):
  q=original/p.name
  if q.exists():
   if q.read_bytes()!=p.read_bytes():raise ValueError('LaunchAgent changed: '+p.name)
   agents.append(q)
 entries=unchanged(ROOT,manifest);free_before=shutil.disk_usage(ROOT.parent).free
 tomb=ROOT.with_name('GreenShaping_verified_deletion_20261022')
 if tomb.exists():raise ValueError('deletion target already exists')
 ROOT.rename(tomb)
 try:unchanged(tomb,manifest)
 except BaseException:
  tomb.rename(ROOT);raise
 for q in agents:q.unlink()
 shutil.rmtree(tomb)
 shutil.rmtree(STAGING)
 receipt=dict(proof,deleted_project=str(ROOT),deleted_staging=str(STAGING),entries=entries,agents_removed=len(agents),completed_epoch=time.time(),free_bytes_before=free_before,free_bytes_after=shutil.disk_usage(ROOT.parent).free)
 receipt['freed_bytes']=receipt['free_bytes_after']-free_before
 path=ROOT.with_name('GreenShaping_backup_receipt_20261022.json');path.write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))

if __name__=='__main__':
 import sys
 cleanup(json.load(sys.stdin))
