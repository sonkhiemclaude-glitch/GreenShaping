"""Bounded four-host functional smoke: direct HTTP, upstream proxy, upstream shaping.

This does not run the published benchmark suite or estimate a privacy budget.
The SSH key is never transferred. Experiment traffic uses private VPC addresses.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shlex
import time
from pathlib import Path

import ec2_runner as r

REMOTE = r.INVENTORY['remote_root']
NODES = r.INVENTORY['nodes']


def parallel(function, nodes):
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        return list(pool.map(function, nodes))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', required=True)
    a = p.parse_args()
    r.require_available(list(NODES))
    if not a.run_name.replace('_', '').isalnum():
        raise ValueError('run-name must use letters, digits and underscores')
    local = r.EVIDENCE / a.run_name
    local.mkdir(exist_ok=False)
    remote = REMOTE + '/runs/' + a.run_name
    state: dict = {'state': 'running', 'phase': 'prepare', 'data_kind': 'synthetic_functional',
                   'energy_measured': False, 'privacy_certified': False, 'run_name': a.run_name}
    containers: list[tuple[str, str]] = []
    captures: list[tuple[str, int, str]] = []
    server_pid = None

    def update(phase: str, **kw) -> None:
        state.update(phase=phase, **kw)
        (local / 'status.json').write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state), flush=True)

    def collect_container(node: str, name: str) -> None:
        output = r.python(node, f'''import json,subprocess
name={name!r}
subprocess.run(['sudo','-n','docker','stop','--time','5',name],capture_output=True)
logs=subprocess.run(['sudo','-n','docker','logs',name],capture_output=True,text=True)
info=subprocess.run(['sudo','-n','docker','inspect',name],capture_output=True,text=True)
subprocess.run(['sudo','-n','docker','rm',name],capture_output=True)
print(json.dumps({{'log_stdout':logs.stdout,'log_stderr':logs.stderr,'inspect':json.loads(info.stdout) if info.returncode==0 else None}}))
''')
        (local / f'{name}_runtime.json').write_text(output)

    def stop_captures() -> None:
        for node, pid, phase in list(captures):
            result = r.python(node, f'''import json,subprocess,time
from pathlib import Path
pid={pid};folder=Path({remote!r})/{phase!r}
command=Path('/proc/'+str(pid)+'/cmdline')
if command.exists() and b'tcpdump' in command.read_bytes() and str(folder).encode() in command.read_bytes():
 subprocess.run(['sudo','-n','kill','-INT','--','-'+str(pid)],capture_output=True)
for _ in range(30):
 text=(folder/'tcpdump.log').read_text()
 if 'packets dropped by kernel' in text:break
 time.sleep(.1)
print(json.dumps({{'log':text}}))
''')
            (local / f'{node}_{phase}_capture.json').write_text(result)
            r.copy(node, local / f'{node}_{phase}.pcap', remote + '/' + phase + '/traffic.pcap', get=True)
            captures.remove((node, pid, phase))

    try:
        parallel(lambda n: r.python(n, f'''from pathlib import Path
p=Path({remote!r});p.mkdir(parents=True,exist_ok=False)
'''), NODES)
        root = Path(__file__).resolve().parent
        for node, files in {'T1': ['ec2_payload_client.py', 'ec2_payload_server.py'],
                            'T4': ['ec2_payload_server.py']}.items():
            for name in files:
                r.copy(node, root / name, REMOTE + '/' + name)
        server_pid = json.loads(r.python('T4', f'''import json,subprocess
from pathlib import Path
root=Path({remote!r})
with (root/'server.log').open('ab') as log:
 args=['python3',{REMOTE + '/ec2_payload_server.py'!r},'--bind',{NODES['T4']['private_ip']!r},'--duration','900']
 p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(json.dumps({{'pid':p.pid}}))
'''))['pid']
        # Check private HTTP connectivity before starting any CPU-intensive middlebox.
        update('private_network_preflight')
        r.python('T1', f'''import http.client
c=http.client.HTTPConnection({NODES['T4']['private_ip']!r},5555,timeout=4)
c.request('GET','/payload/1024?trial=network-preflight')
response=c.getresponse();assert response.status==200 and len(response.read())==1024
c.close()
''')
        certificate = r.python('T3', f'''import subprocess,json,hashlib
from pathlib import Path
folder=Path({remote!r})/'tls';folder.mkdir()
subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','2','-subj','/CN=netshaper-ec2-functional','-keyout',str(folder/'server.key'),'-out',str(folder/'server.cert')],capture_output=True,check=True)
(folder/'server.key').chmod(0o600)
print(json.dumps({{'public_cert_sha256':hashlib.sha256((folder/'server.cert').read_bytes()).hexdigest()}}))
''')
        (local / 'certificate.json').write_text(certificate)
        images = {n: json.loads((r.EVIDENCE / f'{n}_runtime_provenance.json').read_text())['images'] for n in ('T2', 'T3')}
        protocol = {'nodes': NODES, 'images': images, 'source_commit': r.INVENTORY['source_commit'],
                    'provenance': 'upstream binary images pinned by digest; not rebuilt from recorded source',
                    'modes': ['direct', 'no-shaping', 'shaping'], 'sizes': [1024, 16384, 65536, 262144],
                    'repetitions': 3, 'energy_measured': False, 'privacy_certified': False,
                    'limitations': ['synthetic HTTP payloads; functional test only', '8 vCPU/4 physical cores per VM',
                                    'no BIOS control, no isolated CPU cores, no 10Gbps performance claim',
                                    'host Ubuntu 24.04; upstream runtime images Ubuntu 22.04',
                                    'capture includes startup and teardown; not an attacker evaluation',
                                    'noise multiplier parameters do not establish an epsilon claim']}
        (local / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        for mode in ('direct', 'no-shaping', 'shaping'):
            phase = mode.replace('-', '_')
            update(phase)
            def capture(node, phase=phase):
                return node, json.loads(r.python(node, f'''import subprocess,json
from pathlib import Path
folder=Path({remote!r})/{phase!r};folder.mkdir()
flt=('((host 172.31.37.241 or host 172.31.34.237 or host 172.31.38.41 or host 172.31.46.188) '
     'and (tcp port 5555 or tcp port 8000 or udp port 4567))')
with (folder/'tcpdump.log').open('ab') as log:
 p=subprocess.Popen(['sudo','-n','timeout','--signal=INT','--kill-after=3s','90s','tcpdump','-U','-n','-i','ens5','-s','128','-Z','ubuntu','-w',str(folder/'traffic.pcap'),flt],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(json.dumps({{'pid':p.pid}}))
'''))['pid']
            for node, pid in parallel(capture, NODES):
                captures.append((node, pid, phase))
            if mode != 'direct':
                common = {'noiseMultiplier': 1., 'sensitivity': 4096, 'maxDecisionSize': 65536, 'minDecisionSize': 0,
                          'DPCreditorLoopInterval': 50000, 'sendingLoopInterval': 0,
                          'shaperCores': [2], 'workerCores': [3]}
                configs = {
                    'T2': {'logLevel': 'WARNING', 'maxClients': 4, 'queueSize': 2097152,
                           'appName': f'gsnet_{a.run_name}_{phase}_peer1',
                           'shapedClient': dict(common, peer2Addr=NODES['T3']['private_ip'], peer2Port=4567, sendingStrategy='BURST'),
                           'unshapedServer': {'bindAddr': NODES['T2']['private_ip'], 'bindPort': 8000,
                                              'serverAddr': NODES['T4']['private_ip'] + ':5555', 'checkQueuesInterval': 500, 'cores': [4]}},
                    'T3': {'logLevel': 'WARNING', 'maxPeers': 1, 'maxStreamsPerPeer': 4, 'queueSize': 2097152,
                           'appName': f'gsnet_{a.run_name}_{phase}_peer2',
                           'shapedServer': dict(common, serverCert='/tls/server.cert', serverKey='/tls/server.key', listeningPort=4567,
                                                sensitivity=65536, maxDecisionSize=262144, strategy='BURST'),
                           'unshapedClient': {'checkQueuesInterval': 500, 'cores': [4]}}
                }
                for node in ('T3', 'T2'):
                    name = f'gsnet-{a.run_name}-{phase}-{node.lower()}'
                    peer = 'peer2' if node == 'T3' else 'peer1'
                    image = next(v for v in images[node] if v['tag'].endswith(peer + '-' + mode))
                    cfg = configs[node]
                    (local / f'{node}_{phase}_config.json').write_text(json.dumps(cfg, indent=2) + '\n')
                    command = ['sudo', '-n', 'docker', 'run', '-d', '--init', '--name', name, '--network', 'host',
                               '--shm-size=256m', '--memory=2g', '--pids-limit=256',
                               '--mount', f'type=bind,source={remote}/{phase},target=/work,readonly']
                    if node == 'T3':
                        command += ['--mount', f'type=bind,source={remote}/tls,target=/tls,readonly']
                    command += ['--entrypoint', '/usr/bin/timeout', image['id'], '--signal=INT', '--kill-after=5s',
                                '90s', '/root/peer_2' if node == 'T3' else '/root/peer_1', '/work/config.json']
                    r.python(node, f'''import subprocess,json
from pathlib import Path
(Path({remote!r})/{phase!r}/'config.json').write_text(json.dumps({cfg!r},indent=2))
p=subprocess.run({command!r},capture_output=True,text=True,check=True)
print(p.stdout)
''')
                    containers.append((node, name))
                time.sleep(5)
                for node, name in containers:
                    r.python(node, f'''import json,subprocess
v=json.loads(subprocess.check_output(['sudo','-n','docker','inspect',{name!r}],text=True))[0]
assert v['State']['Running'],v['State']
''')
            host = NODES['T4' if mode == 'direct' else 'T2']['private_ip']
            port = 5555 if mode == 'direct' else 8000
            cmd = shlex.join(['python3', REMOTE + '/ec2_payload_client.py', '--host', host, '--port', str(port),
                              '--mode', mode, '--out', remote + '/' + phase + '/client_results.json'])
            result = r.run('T1', cmd, timeout=65)
            (local / f'{phase}_client.log').write_text(result.stdout + result.stderr)
            r.copy('T1', local / f'{phase}_client_results.json', remote + '/' + phase + '/client_results.json', get=True)
            if result.returncode:
                raise RuntimeError(f'{phase} client failed; inspect retained results and container logs')
            while containers:
                node, name = containers.pop()
                collect_container(node, name)
            stop_captures()
        r.copy('T4', local / 'server.log', remote + '/server.log', get=True)
        server_rows = [json.loads(line) for line in (local / 'server.log').read_text().splitlines() if line.startswith('{')]
        for mode in ('direct', 'no-shaping', 'shaping'):
            source = NODES['T1' if mode == 'direct' else 'T3']['private_ip']
            matched = [v for v in server_rows if '?trial=' + mode + '-' in v['path']]
            if len(matched) != 12 or any(v['source_ip'] != source for v in matched):
                raise RuntimeError(f'unexpected server source/path/count for {mode}')
        hashes = {str(f.relative_to(local)): hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in local.rglob('*') if f.is_file() and f.name != 'status.json'}
        (local / 'manifest.json').write_text(json.dumps(hashes, indent=2) + '\n')
        update('complete', state='complete', successful_requests=36, server_sources_verified=True)
        return 0
    except Exception as e:
        update(state.get('phase', 'unknown'), state='failed', error=str(e))
        raise
    finally:
        while containers:
            node, name = containers.pop()
            try:
                collect_container(node, name)
            except Exception as e:
                print(f'cleanup {node}/{name}: {e}', flush=True)
        if captures:
            try:
                stop_captures()
            except Exception as e:
                print(f'capture cleanup: {e}', flush=True)
        if server_pid is not None:
            r.python('T4', f'''import os,signal
from pathlib import Path
p=Path('/proc/{server_pid}/cmdline')
if p.exists() and b'ec2_payload_server.py' in p.read_bytes():os.kill({server_pid},signal.SIGTERM)
''')


if __name__ == '__main__':
    raise SystemExit(main())
