"""Prepare on Mac Studio now; install/start a bounded batch only after EC2 restart.

The original SonJ.pem remains on the controller. A separate expiring evaluation
key is generated on Mac Studio. All access requires existing pinned host keys.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import ec2_runner as r

from greenshaping.utils.evidence import atomic_json
from greenshaping.utils.provenance import file_sha256


def mac(operation, *args, data=None, timeout=90):
    command = [sys.executable, str(r.ROOT / 'scripts/remote_runner.py'), '--known-hosts',
               str(r.ROOT / 'outputs/remote_20261020/known_hosts'), operation, *map(str, args)]
    result = subprocess.run(command, input=data, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError('Mac staging failed: ' + result.stderr[-1500:])
    return result.stdout


def execute(source, timeout=90):
    interpreter = '/Users/sonha/Projects/GreenShaping/run_v4_20261020_r2/.venv/bin/python'
    return mac('exec', interpreter + ' -', '--timeout', str(timeout - 5), data=source, timeout=timeout)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--install-and-start', action='store_true', help='requires explicitly released EC2 host reservations')
    p.add_argument('--max-seconds', type=int, default=2700)
    a = p.parse_args()
    if not re.fullmatch('[A-Za-z0-9_]+', a.run_id) or not 300 <= a.max_seconds <= 2700:
        p.error('safe run ID and a 5..45 minute window required')
    a.out.mkdir(parents=True, exist_ok=True)
    root = '/Users/sonha/Projects/GreenShaping/cloud_runs/' + a.run_id
    app = root + '/app'
    access = root + '/access'
    py = '/Users/sonha/Projects/GreenShaping/run_v4_20261020_r2/.venv/bin/python'
    proof = a.out / 'prepared.json'
    if not proof.exists():
        execute(f'''from pathlib import Path
import subprocess
R=Path({root!r});R.mkdir(parents=True,exist_ok=False);(R/'access').mkdir(mode=0o700)
subprocess.run(['ssh-keygen','-q','-t','ed25519','-N','','-C',{'gsbounded_'+a.run_id!r},'-f',str(R/'access/id_ed25519')],check=True)
''')
        mac('put', a.runtime, root + '/runtime.tar.gz', timeout=180)
        execute(f'''from pathlib import Path
import tarfile,hashlib
R=Path({root!r});p=R/'runtime.tar.gz'
assert hashlib.sha256(p.read_bytes()).hexdigest()=={file_sha256(a.runtime)!r}
(R/'app').mkdir()
with tarfile.open(p) as tar:tar.extractall(R/'app',filter='data')
''')
        atomic_json(proof, {'run_id': a.run_id, 'mac_root': root, 'runtime_sha256': file_sha256(a.runtime),
                            'cloud_started': False, 'SonJ_transferred': False})
    elif json.loads(proof.read_text())['runtime_sha256'] != file_sha256(a.runtime):
        raise ValueError('prepared source changed; use a new run ID')
    if not a.install_and_start:
        print(json.dumps({'prepared': True, 'cloud_contacted': False, 'mac_root': root}))
        return
    execute(f'''import shutil
assert shutil.disk_usage({root!r}).free >= 10*1024**3, 'Mac Studio needs >=10 GiB free before cloud capture'
''')
    r.require_available(list(r.INVENTORY['nodes']))
    if (a.out / 'launch.json').exists():
        raise RuntimeError('launch already recorded; review before any replay')
    deadline = time.time() + a.max_seconds
    remote = r.INVENTORY['remote_root'] + '/runs/' + a.run_id
    public = execute(f'from pathlib import Path\nprint((Path({access!r})/"id_ed25519.pub").read_text().strip())').strip()
    if not re.fullmatch(r'ssh-ed25519 [A-Za-z0-9+/=]+ gsbounded_' + re.escape(a.run_id), public):
        raise ValueError('unexpected evaluation public key')
    for node in r.INVENTORY['nodes']:
        r.python(node, f'from pathlib import Path\nPath({remote!r}).mkdir(parents=True,exist_ok=False)')
        r.copy(node, a.runtime, remote + '/runtime.tar.gz')
        gate = f'python3 {remote}/app/scripts/bounded_ssh_gate.py --deadline-epoch {deadline+90}'
        line = f'restrict,command="{gate}" ' + public
        r.python(node, f'''import tarfile,os,hashlib
from pathlib import Path
R=Path({remote!r});(R/'app').mkdir()
assert hashlib.sha256((R/'runtime.tar.gz').read_bytes()).hexdigest()=={file_sha256(a.runtime)!r}
with tarfile.open(R/'runtime.tar.gz') as tar:tar.extractall(R/'app',filter='data')
p=Path.home()/'.ssh/authorized_keys'
with p.open('a') as f:f.write({chr(10)!r}+{line!r}+{chr(10)!r});f.flush();os.fsync(f.fileno())
p.chmod(0o600)
''')
        if node in ('T2', 'T3'):
            r.python(node, '''import shutil,subprocess
if not shutil.which('ethtool'):
 subprocess.run(['sudo','-n','apt-get','-o','DPkg::Lock::Timeout=30','install','-y','ethtool'],
                check=True,timeout=90)
''', timeout=100)
    atomic_json(a.out / 'inventory.json', r.INVENTORY)
    atomic_json(a.out / 'host_leases.json', {n: {'reserved': False, 'available_until_epoch': deadline,
        'reason': 'User restarted hosts for one bounded TRAIN capture calibration'} for n in r.INVENTORY['nodes']})
    for source, target in [(a.out / 'inventory.json', access + '/inventory.json'),
            (a.out / 'host_leases.json', access + '/host_leases.json'),
            (r.EVIDENCE / 'known_hosts', access + '/known_hosts'),
            *[(r.EVIDENCE / f'{n}_runtime_provenance.json', access + f'/{n}_runtime_provenance.json') for n in ['T2', 'T3']]]:
        mac('put', source, target)
    env = {'GREENSHAPING_EC2_IDENTITY': access + '/id_ed25519',
           'GREENSHAPING_EC2_KNOWN_HOSTS': access + '/known_hosts', 'GREENSHAPING_EC2_INVENTORY': access + '/inventory.json',
           'GREENSHAPING_EC2_RESERVATIONS': access + '/host_leases.json', 'GREENSHAPING_EC2_EVIDENCE': access,
           'PYTHONPATH': app + '/src', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    # This read-only live gate catches different public IPs, SSH SG rules and
    # forced-command/SFTP problems before a measurement queue is launched.
    execute(f'''import sys,os,json
from pathlib import Path
os.environ.update({env!r});sys.path.insert(0,{app+'/scripts'!r});import ec2_runner as r
r.require_available(list(r.INVENTORY['nodes']))
probe="from pathlib import Path;print(Path('/proc/sys/kernel/random/boot_id').read_text())"
result={{n:r.python(n,probe).strip() for n in r.INVENTORY['nodes']}}
import hashlib
local=Path({root!r})/'transfer_probe.txt';local.write_bytes(b'greenshaping-sftp-roundtrip-v1')
for n in r.INVENTORY['nodes']:
 target={remote!r}+'/transfer_probe.txt'
 r.copy(n,local,target)
 received=Path({root!r})/(n+'_transfer_probe.txt');r.copy(n,received,target,get=True)
 assert received.read_bytes()==local.read_bytes(), 'SFTP roundtrip mismatch: '+n
Path({root+'/ssh_preflight.json'!r}).write_text(json.dumps({{'boots':result,'sftp_roundtrip_all_passed':True}},indent=2))
''', timeout=120)
    job = {'id': 'matched_capture', 'argv': [py, app + '/scripts/run_ec2_matched_capture.py', '--out', root + '/results',
        '--remote-root', remote, '--deadline-epoch', str(deadline)], 'cwd': app, 'env': env,
        'timeout_s': max(1, int(deadline - time.time())), 'safe_to_retry': False, 'max_attempts': 1}
    result = execute(f'''import json,time,subprocess,plistlib,os
from pathlib import Path
root=Path({root!r});Q=root/'queue';Q.mkdir()
(Q/'queue.json').write_text(json.dumps({{'deadline_epoch':{deadline},'poll_s':2,'exit_when_idle':True,'jobs':[{job!r}]}}))
label={'org.greenshaping.cloud.'+a.run_id!r};plist=Path.home()/'Library/LaunchAgents'/(label+'.plist')
plist.parent.mkdir(parents=True,exist_ok=True)
plist.write_bytes(plistlib.dumps({{'Label':label,'ProgramArguments':[{py!r},{app+'/scripts/durable_worker.py'!r},'--root',str(Q)],
'RunAtLoad':True,'StandardOutPath':str(Q/'launchd.stdout.log'),'StandardErrorPath':str(Q/'launchd.stderr.log')}}))
p=subprocess.run(['launchctl','bootstrap','gui/'+str(os.getuid()),str(plist)],capture_output=True,text=True)
if p.returncode:raise RuntimeError(p.stderr)
print(json.dumps({{'queue':str(Q),'deadline_epoch':{deadline},'launch_agent':label}}))
''')
    atomic_json(a.out / 'launch.json', {'mac': json.loads(result), 'remote_root': remote, 'cloud_started': True,
        'epoch': time.time(), 'user_managed_instance_stop': True})
    print(result)


if __name__ == '__main__':
    main()
