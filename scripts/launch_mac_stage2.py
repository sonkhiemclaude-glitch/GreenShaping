"""Prepare local runtime/plan; --install deploys only to pinned Tailscale Mac.

No cloud access. Storage guard runs before creating a Mac project directory.
"""
import argparse
import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path
from launch_mac_cloud_batch import execute, mac
from greenshaping.utils.evidence import atomic_json
from greenshaping.utils.provenance import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--install',action='store_true');a=p.parse_args()
    app=Path(__file__).resolve().parents[1];out=app/'outputs/stage2_preparation_20261021';out.mkdir(exist_ok=True)
    runtime=out/'runtime_stage2_09.tar.gz'
    if not runtime.exists():
        with tarfile.open(runtime,'w:gz') as tar:
            for folder in ['src','scripts','tests']:
                for f in sorted((app/folder).rglob('*.py')):
                    if '__pycache__' not in f.parts:tar.add(f,arcname=str(f.relative_to(app)))
            tar.add(app/'pyproject.toml',arcname='pyproject.toml')
            tar.add(app/'outputs/synthetic_demo',arcname='outputs/synthetic_demo')
            # Offline test imports need inventory metadata, but no real cloud
            # addresses, identity or permission is provisioned in this profile.
            offline={'remote_root':'/offline/no-cloud','nodes':{n:{'public_ip':'127.0.0.1','private_ip':'127.0.0.1','role':'offline'} for n in ['T1','T2','T3','T4']}}
            reservations={n:{'reserved':True,'available_until_epoch':0,'reason':'Mac-only offline profile; cloud disabled'} for n in offline['nodes']}
            for name,value in [('ec2_netshaper_20261020.json',offline),('ec2_host_reservations.json',reservations)]:
                data=json.dumps(value,indent=2).encode();info=tarfile.TarInfo('configs/'+name);info.size=len(data);info.mode=0o600
                tar.addfile(info,io.BytesIO(data))
        atomic_json(out/'runtime_stage2_09.sha256.json',{'sha256':file_sha256(runtime)})
    with tarfile.open(runtime) as tar:
        for m in tar:
            if m.isfile() and m.name.endswith('.py'):
                if hashlib.sha256(tar.extractfile(m).read()).hexdigest()!=file_sha256(app/m.name):
                    raise RuntimeError('Frozen runtime differs from current source; prepare a new runtime ID')
    root='/Users/sonha/Projects/GreenShaping/stage2_preparation_20261021_09'
    py='/Users/sonha/Projects/GreenShaping/run_v4_20261020_r2/.venv/bin/python'
    data='/Users/sonha/Projects/GreenShaping/large_preparation_20261021/evidence'
    plan={'cloud_used':False,'final_test_allowed':False,'mac_root':root,'runtime_sha256':file_sha256(runtime),
          'required_run_free_GiB':4,'profile':'JSON-only bounded preparation; no pcap recording or model downloads','pipeline_output_cap_MiB':512,'max_storage_wait_seconds':14400,'max_execution_seconds':10800,
          'install_argv':['.venv/bin/python','scripts/launch_mac_stage2.py','--install']}
    atomic_json(out/'deployment_plan.json',plan)
    if not a.install:print(json.dumps(plan));return
    launch_record=out/('launch_'+Path(root).name+'.json')
    if launch_record.exists():raise RuntimeError('already launched; review instead of replay')
    execute(f'''import shutil
from pathlib import Path
assert shutil.disk_usage('/Users/sonha/Projects/GreenShaping').free >= 1024**3, 'Mac needs >=1 GiB even for staging; no write attempted'
R=Path({root!r});R.mkdir(exist_ok=False)
''')
    mac('put',runtime,root+'/runtime.tar.gz',timeout=180)
    execute(f'''from pathlib import Path
import tarfile,hashlib
R=Path({root!r});assert hashlib.sha256((R/'runtime.tar.gz').read_bytes()).hexdigest()=={file_sha256(runtime)!r}
(R/'app').mkdir()
with tarfile.open(R/'runtime.tar.gz') as t:t.extractall(R/'app',filter='data')
''')
    job={'id':'mac_only_stage2','argv':['/usr/bin/caffeinate','-i','-s',py,root+'/app/scripts/run_mac_stage2_preparation.py','--data',data,'--out',root+'/results'],
         'cwd':root+'/app','env':{'PYTHONPATH':root+'/app/src','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},
         'timeout_s':25200,'safe_to_retry':False,'max_attempts':1}
    result=execute(f'''from pathlib import Path
import os,json,plistlib,subprocess,time
R=Path({root!r});Q=R/'queue';Q.mkdir()
(Q/'queue.json').write_text(json.dumps({{'deadline_epoch':time.time()+25200,'poll_s':5,'exit_when_idle':True,'jobs':[{job!r}]}}))
label='org.greenshaping.mac.stage2.20261021.09'
p=Path.home()/'Library/LaunchAgents'/(label+'.plist');p.parent.mkdir(exist_ok=True,parents=True)
p.write_bytes(plistlib.dumps({{'Label':label,'ProgramArguments':[{py!r},{root+'/app/scripts/durable_worker.py'!r},'--root',str(Q)],
 'RunAtLoad':True,'ProcessType':'Interactive','Nice':0,'LowPriorityIO':False,'StandardOutPath':str(Q/'launchd.stdout.log'),'StandardErrorPath':str(Q/'launchd.stderr.log')}}))
x=subprocess.run(['launchctl','bootstrap','gui/'+str(os.getuid()),str(p)],capture_output=True,text=True)
if x.returncode:raise RuntimeError(x.stderr)
print(json.dumps({{'queue':str(Q),'launch_agent':label,'cloud_used':False}}))
''')
    atomic_json(launch_record,json.loads(result));atomic_json(out/'latest_launch.json',json.loads(result));print(result)


if __name__=='__main__':main()
