"""Encrypted complete project snapshot, split into bounded parts; no deletion."""
import argparse,hashlib,json,os,stat,struct,tarfile,time,io,plistlib,subprocess,gzip
from pathlib import Path
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes


def atomic(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.chmod(t,0o600);t.replace(p)


class Parts:
 def __init__(self,out,limit=256*1024**2):self.out=out;self.limit=limit;self.f=None;self.n=0;self.used=0;self.rows=[];self.sha=None;self.md5=None
 def write(self,b):
  original=len(b)
  while b:
   if self.f is None:
    self.n+=1;self.path=self.out/('GreenShaping_full.aesgcm.part%03d'%self.n);self.f=self.path.open('xb');os.chmod(self.path,0o600);self.used=0;self.sha=hashlib.sha256();self.md5=hashlib.md5()
   piece=b[:self.limit-self.used];self.f.write(piece);self.sha.update(piece);self.md5.update(piece);self.used+=len(piece);b=b[len(piece):]
   if self.used==self.limit:self.closepart()
  return original
 def closepart(self):
  if self.f:
   self.f.flush();os.fsync(self.f.fileno());self.f.close();self.rows.append({'name':self.path.name,'bytes':self.used,'sha256':self.sha.hexdigest(),'md5':self.md5.hexdigest()});self.f=None
 def flush(self):
  if self.f:self.f.flush()
 def close(self):self.closepart()


class Encrypted:
 def __init__(self,parts,pub):
  self.parts=parts;key=os.urandom(32);nonce=os.urandom(12)
  wrapped=pub.encrypt(key,padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),algorithm=hashes.SHA256(),label=None))
  import base64
  header=json.dumps({'version':1,'cipher':'AES-256-GCM','wrapped_key':base64.b64encode(wrapped).decode(),'nonce':base64.b64encode(nonce).decode(),'compression':'tar.gz'},sort_keys=True).encode()
  prefix=b'GSFULL1\n'+struct.pack('!I',len(header))+header;self.parts.write(prefix)
  self.enc=Cipher(algorithms.AES(key),modes.GCM(nonce)).encryptor();self.enc.authenticate_additional_data(prefix)
 def write(self,b):self.parts.write(self.enc.update(b));return len(b)
 def flush(self):self.parts.flush()
 def finish(self):self.parts.write(self.enc.finalize());self.parts.write(self.enc.tag);self.parts.close()


class HashReader:
 def __init__(self,f,deadline):self.f=f;self.h=hashlib.sha256();self.deadline=deadline
 def read(self,n=-1):
  if time.monotonic()>self.deadline:raise TimeoutError('snapshot deadline')
  b=self.f.read(n);self.h.update(b);return b


def snapshot(root,out,public,controls=None):
 out.mkdir(exist_ok=False);payload=out/'payload';payload.mkdir();deadline=time.monotonic()+1800
 atomic(out/'status.json',{'stage':'snapshotting','root':str(root),'deletion_performed':False})
 pub=serialization.load_pem_public_key(Path(public).read_bytes());parts=Parts(payload);encrypted=Encrypted(parts,pub);manifest={'root':str(root),'created_epoch':time.time(),'entries':{}}
 paths=[]
 for d,dirs,files in os.walk(root,followlinks=False):
  for n in sorted(dirs+files):paths.append(Path(d)/n)
 paths.sort(key=lambda p:str(p.relative_to(root)))
 try:
  with gzip.GzipFile(fileobj=encrypted,mode='wb',compresslevel=3) as compressed, tarfile.open(fileobj=compressed,mode='w|',dereference=False) as tar:
   for i,p in enumerate(paths):
    rel=str(p.relative_to(root));st=p.lstat();entry={'mode':stat.S_IMODE(st.st_mode),'mtime_ns':st.st_mtime_ns}
    # Store each regular file independently, even if source uses hard links.
    tar.inodes={};info=tar.gettarinfo(str(p),arcname='GreenShaping/'+rel)
    if stat.S_ISREG(st.st_mode):
     with p.open('rb') as f:
      reader=HashReader(f,deadline);tar.addfile(info,reader);checksum=reader.h.hexdigest()
     now=p.stat()
     if (st.st_size,st.st_mtime_ns,st.st_ino)!=(now.st_size,now.st_mtime_ns,now.st_ino):raise RuntimeError('file changed during snapshot: '+rel)
     entry.update(type='file',bytes=st.st_size,sha256=checksum)
    elif stat.S_ISLNK(st.st_mode):entry.update(type='symlink',target=os.readlink(p));tar.addfile(info)
    elif stat.S_ISDIR(st.st_mode):entry.update(type='directory');tar.addfile(info)
    else:raise ValueError('unsupported special file: '+rel)
    manifest['entries'][rel]=entry
    if i%1000==0:atomic(out/'status.json',{'stage':'snapshotting','completed_entries':i,'total_entries':len(paths),'deletion_performed':False})
   if controls:
    tar.add(str(controls),arcname='MacControl',recursive=True)
   data=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('BACKUP_MANIFEST.json');info.size=len(data);info.mode=0o600;tar.addfile(info,io.BytesIO(data))
  encrypted.finish()
  atomic(out/'manifest.json',manifest)
  atomic(payload/'PARTS.json',{'format':'GSFULL1: concatenate numbered parts, then RSA-OAEP/AES-GCM decrypt to tar.gz','parts':parts.rows,'manifest_sha256':hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),'entries':len(paths),'source_file_bytes':sum(v.get('bytes',0) for v in manifest['entries'].values())})
  atomic(out/'status.json',{'stage':'snapshot_complete','parts':len(parts.rows),'archive_bytes':sum(r['bytes'] for r in parts.rows),'source_bytes':sum(v.get('bytes',0) for v in manifest['entries'].values()),'entries':len(paths),'deletion_performed':False})
 except BaseException as e:
  parts.close();atomic(out/'status.json',{'stage':'snapshot_failed','error':repr(e),'deletion_performed':False});raise


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--public-key',type=Path,required=True);p.add_argument('--controls',type=Path);a=p.parse_args()
 if a.root.resolve() in a.out.resolve().parents:raise ValueError('staging must be outside project')
 snapshot(a.root,a.out,a.public_key,a.controls)

if __name__=='__main__':main()
