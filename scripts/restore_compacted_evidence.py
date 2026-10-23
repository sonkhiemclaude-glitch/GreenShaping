"""Restore one archived raw copy without overwriting or deleting evidence."""
import argparse
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def restore(row,destination):
    archive=Path(row['archive']);destination=Path(destination)
    if destination.exists():
        if sha(destination)==row['sha256']:return
        raise FileExistsError('different file exists; no overwrite')
    if sha(archive)!=row['archive_sha256']:raise ValueError('archive checksum mismatch')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(destination.parent).free<row['size']+512*1024**2:raise RuntimeError('insufficient restore space')
    temp=destination.with_name(destination.name+'.restoring')
    with tarfile.open(archive) as tar:
        member=tar.getmember(row['member'])
        if not member.isfile() or member.size!=row['size']:raise ValueError('member type/size mismatch')
        with temp.open('xb') as f,tar.extractfile(member) as source:
            shutil.copyfileobj(source,f,1024**2);f.flush();os.fsync(f.fileno())
    if sha(temp)!=row['sha256']:raise ValueError('restored checksum mismatch; partial retained')
    os.replace(temp,destination)
    fd=os.open(destination.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True)
    p.add_argument('--entry',type=int,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    row=json.loads(a.index.read_text())[a.entry];restore(row,a.destination)
    print(json.dumps({'restored':str(a.destination),'sha256':row['sha256']}))


if __name__=='__main__':main()
