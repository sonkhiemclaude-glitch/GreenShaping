import importlib.util,sys
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

def load(n):
 p=Path(__file__).parents[1]/'scripts'/(n+'.py');s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_complete_encrypted_snapshot_restores_and_detects_tampering(tmp_path, monkeypatch):
 import pytest,os
 pack=load('mac_project_full_snapshot');restore=load('verify_mac_full_backup')
 original=pack.Parts
 monkeypatch.setattr(pack,'Parts',lambda out:original(out,limit=16000))
 root=tmp_path/'root';root.mkdir();(root/'sub').mkdir();(root/'sub/secret.key').write_bytes(b'secret test fixture');(root/'data').write_bytes(os.urandom(100000));(root/'alias').symlink_to('data')
 key=rsa.generate_private_key(public_exponent=65537,key_size=2048);pub=tmp_path/'public.pem';priv=tmp_path/'private.pem'
 pub.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo));priv.write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
 out=tmp_path/'snapshot';pack.snapshot(root,out,pub)
 result=restore.verify(out/'payload',priv,tmp_path/'restored')
 assert result['authenticated'] and result['entries']==4 and result['parts']>1
 assert (tmp_path/'restored/GreenShaping/sub/secret.key').read_bytes()==b'secret test fixture'
 assert (tmp_path/'restored/GreenShaping/alias').is_symlink()
 part=next((out/'payload').glob('*.part*'));b=bytearray(part.read_bytes());b[-20]^=1;part.write_bytes(b)
 with pytest.raises(ValueError,match='checksum'):restore.verify(out/'payload',priv)

def test_cleanup_refuses_changed_and_added_files(tmp_path):
 import pytest,json
 cleanup=load('mac_backup_cleanup');pack=load('mac_project_full_snapshot')
 root=tmp_path/'root';root.mkdir();(root/'file').write_bytes(b'original')
 key=rsa.generate_private_key(public_exponent=65537,key_size=2048);pub=tmp_path/'public.pem';pub.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo))
 pack.snapshot(root,tmp_path/'snapshot',pub);manifest=json.loads((tmp_path/'snapshot/manifest.json').read_text())
 assert cleanup.unchanged(root,manifest)==1
 (root/'new').write_text('new')
 with pytest.raises(ValueError,match='inventory'):cleanup.unchanged(root,manifest)
 (root/'new').unlink();(root/'file').write_bytes(b'modified')
 with pytest.raises(ValueError,match='changed'):cleanup.unchanged(root,manifest)
 assert root.exists()

def test_direct_upload_resumes_acknowledged_chunks(tmp_path,monkeypatch):
 import io,json,urllib.error
 worker=load('mac_drive_upload_worker');monkeypatch.setattr(worker,'ROOT',tmp_path);monkeypatch.setattr(worker,'CHUNK_BYTES',16)
 p=tmp_path/'snapshot02/payload';p.mkdir(parents=True);data=b'x'*19;(p/'part').write_bytes(data);seen=[]
 class Response(io.BytesIO):
  def __enter__(self):return self
  def __exit__(self,*a):self.close()
 def request(req,timeout):
  seen.append((req.get_header('Content-range'),len(req.data)))
  if len(seen)==1:raise urllib.error.HTTPError(req.full_url,308,'Resume',{},None)
  if len(seen)==2:raise urllib.error.HTTPError(req.full_url,308,'Resume',{'Range':'bytes=0-15'},None)
  return Response(json.dumps({'id':'fixture-id'}).encode())
 monkeypatch.setattr(worker.urllib.request,'urlopen',request)
 r=worker.upload({'name':'part','bytes':len(data),'url':'https://example.invalid/scoped-fixture'})
 assert r['completed'] and seen==[('bytes */19',0),('bytes 0-15/19',16),('bytes 16-18/19',3)]
