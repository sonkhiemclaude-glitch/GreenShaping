"""Verify or restore a GSFULL1 encrypted multipart project archive."""
import argparse,base64,gzip,hashlib,json,os,struct,tarfile
from pathlib import Path
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes


class Concat:
 def __init__(self,files):self.files=iter(files);self.f=None
 def read(self,n):
  result=bytearray()
  while len(result)<n:
   if self.f is None:
    try:self.f=next(self.files).open('rb')
    except StopIteration:break
   b=self.f.read(n-len(result));result.extend(b)
   if not b:self.f.close();self.f=None
  return bytes(result)


class Decrypted:
 def __init__(self,stream,total,private):
  prefix=stream.read(12)
  if prefix[:8]!=b'GSFULL1\n':raise ValueError('invalid archive magic')
  n=struct.unpack('!I',prefix[8:])[0]
  if not 1<n<65536:raise ValueError('bad header length')
  h=stream.read(n);meta=json.loads(h);key=private.decrypt(base64.b64decode(meta['wrapped_key']),padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),algorithm=hashes.SHA256(),label=None))
  self.dec=Cipher(algorithms.AES(key),modes.GCM(base64.b64decode(meta['nonce']))).decryptor();self.dec.authenticate_additional_data(prefix+h)
  self.stream=stream;self.left=total-12-n-16;self.authenticated=False
 def read(self,n=-1):
  if n<0:n=1024**2
  if self.left:
   b=self.stream.read(min(n,self.left))
   if not b:raise ValueError('truncated ciphertext')
   self.left-=len(b);return self.dec.update(b)
  if not self.authenticated:
   tag=self.stream.read(16)
   if len(tag)!=16:raise ValueError('missing authentication tag')
   self.dec.finalize_with_tag(tag);self.authenticated=True
   if self.stream.read(1):raise ValueError('unexpected trailing bytes')
  return b''


def verify(parts,private_key,restore_to=None):
 meta=json.loads((parts/'PARTS.json').read_text());files=[]
 for row in meta['parts']:
  p=parts/row['name'];h=hashlib.sha256()
  if p.stat().st_size!=row['bytes']:raise ValueError('part size mismatch')
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024**2),b''):h.update(b)
  if h.hexdigest()!=row['sha256']:raise ValueError('part checksum mismatch: '+p.name)
  files.append(p)
 private=serialization.load_pem_private_key(private_key.read_bytes(),password=None)
 dec=Decrypted(Concat(files),sum(p.stat().st_size for p in files),private)
 if restore_to:restore_to.mkdir(exist_ok=False)
 observed={};manifest=None
 with gzip.GzipFile(fileobj=dec,mode='rb') as gz:
  with tarfile.open(fileobj=gz,mode='r|') as tar:
   for m in tar:
    path=Path(m.name)
    if path.is_absolute() or '..' in path.parts:raise ValueError('unsafe member path')
    target=restore_to/path if restore_to else None
    if target:
     for parent in target.parents:
      if parent==restore_to.parent:break
      if parent.is_symlink():raise ValueError('symlink parent')
    if m.isfile():
     f=tar.extractfile(m);h=hashlib.sha256();capture=bytearray() if m.name=='BACKUP_MANIFEST.json' else None
     dest=None
     if target:target.parent.mkdir(parents=True,exist_ok=True);dest=target.open('xb')
     try:
      for b in iter(lambda:f.read(1024**2),b''):
       h.update(b)
       if capture is not None:capture.extend(b)
       if dest:dest.write(b)
     finally:
      if dest:dest.close();target.chmod(m.mode)
     if capture is not None:
      if hashlib.sha256(capture).hexdigest()!=meta['manifest_sha256']:raise ValueError('manifest hash mismatch')
      manifest=json.loads(capture)
     if m.name.startswith('GreenShaping/'):observed[m.name[len('GreenShaping/'):]]={'type':'file','bytes':m.size,'sha256':h.hexdigest()}
    elif m.issym():
     if target:target.parent.mkdir(parents=True,exist_ok=True);os.symlink(m.linkname,target)
     if m.name.startswith('GreenShaping/'):observed[m.name[len('GreenShaping/'):]]={'type':'symlink','target':m.linkname}
    elif m.isdir():
     if target:target.mkdir(parents=True,exist_ok=True)
     if m.name.startswith('GreenShaping/'):observed[m.name[len('GreenShaping/'):]]={'type':'directory'}
    else:raise ValueError('unexpected archive member type')
  while gz.read(1024**2):pass
 if not dec.authenticated:dec.read(1)
 if not manifest or set(manifest['entries'])!=set(observed):raise ValueError('inventory mismatch')
 for rel,entry in observed.items():
  if any(manifest['entries'][rel].get(k)!=v for k,v in entry.items()):raise ValueError('restored content mismatch: '+rel)
 if restore_to:
  for rel,entry in sorted(manifest['entries'].items(),key=lambda x:len(Path(x[0]).parts),reverse=True):
   p=restore_to/'GreenShaping'/rel
   if entry['type']!='symlink':p.chmod(entry['mode']);os.utime(p,ns=(entry['mtime_ns'],entry['mtime_ns']))
 return {'verified':True,'authenticated':dec.authenticated,'entries':len(observed),'regular_files':sum(e['type']=='file' for e in observed.values()),'source_bytes':meta['source_file_bytes'],'parts':len(files),'restored_to_disk':restore_to is not None}


def main():
 p=argparse.ArgumentParser();p.add_argument('--parts',type=Path,required=True);p.add_argument('--private-key',type=Path,required=True);p.add_argument('--restore-to',type=Path);a=p.parse_args();print(json.dumps(verify(a.parts,a.private_key,a.restore_to)))

if __name__=='__main__':main()
