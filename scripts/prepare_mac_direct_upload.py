import configparser,json,pathlib,urllib.request,urllib.parse,time,urllib.error
BASE=pathlib.Path(__file__).resolve().parents[1];OUT=BASE/'outputs/mac_full_backup_20261022';PARENT='1nQ4JSto8DGKQlVlW8sW_J3uyKgP9hWnJ';NAME='MacStudio_GreenShaping_full_20261022'
def api(path,data=None,extra=None):
 c=configparser.ConfigParser();c.read(pathlib.Path.home()/'.config/rclone/rclone.conf');token=json.loads(c['gdrive']['token'])['access_token']
 h={'Authorization':'Bearer '+token,'Content-Type':'application/json'};h.update(extra or {})
 req=urllib.request.Request('https://www.googleapis.com/'+path,data=json.dumps(data).encode() if data is not None else None,headers=h)
 for i in range(10):
  try:
   with urllib.request.urlopen(req,timeout=60) as r:return dict(r.headers),r.read()
  except urllib.error.HTTPError as e:
   if e.code not in (403,429,500,502,503,504):raise RuntimeError('Drive API HTTP '+str(e.code)) from None
   if i==9:raise RuntimeError('Drive API quota/retry limit') from None
   time.sleep(10)
q="'%s' in parents and name='%s' and trashed=false"%(PARENT,NAME)
_,body=api('drive/v3/files?'+urllib.parse.urlencode({'q':q,'fields':'files(id,name)','pageSize':100}));folders=json.loads(body)['files']
if len(folders)!=1:raise ValueError('expected exactly one destination folder')
folder=folders[0]['id'];(OUT/'direct_folder.json').write_text(json.dumps({'id':folder,'url':'https://drive.google.com/drive/folders/'+folder}))
meta=json.loads((OUT/'payload/PARTS.json').read_text());rows=[]
for r in meta['parts']:
 h,_=api('upload/drive/v3/files?uploadType=resumable&fields=id,name,size,md5Checksum',{'name':r['name'],'parents':[folder]}, {'X-Upload-Content-Type':'application/octet-stream','X-Upload-Content-Length':str(r['bytes'])})
 row=dict(r,url=h.get('Location') or h.get('location'));assert row['url'];rows.append(row)
 print('authorized_part',len(rows),flush=True)
p=OUT/'private/upload_sessions.json';p.write_text(json.dumps(rows));p.chmod(0o600)
print('sessions_ready',len(rows))
