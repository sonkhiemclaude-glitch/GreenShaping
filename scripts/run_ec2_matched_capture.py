"""Finite Mac Studio coordinator for TRAIN four-role matched observer checks.

Requires staged code, current pinned SSH inventory and an approved host lease.
--plan-only never contacts EC2. No final-test, energy or matched-epsilon claim.
"""
import argparse
import concurrent.futures
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

import ec2_runner as r
from ec2_readiness import PROBE, assess

from greenshaping.data.wire_capture import audit
from greenshaping.utils.evidence import SessionLedger, atomic_json
from greenshaping.utils.provenance import file_sha256

PEERS = ('172.31.34.237', '172.31.38.41')
BPF = f'host {PEERS[0]} and host {PEERS[1]} and (tcp or udp)'


def plan():
    modes = ['none', 'pad', 'rounds', 'dp', 'combo', 'netshaper_no_shaping', 'netshaper_shaping']
    random.Random(20261021).shuffle(modes)
    return {'modes': modes, 'questions': 12, 'repetitions': 2, 'planned_sessions': 168,
            'observer_filter': BPF, 'N': 20, 'physical_data_sites': 1,
            'management_plane': 'T1 to T4:8000; outside declared T2-T3 observer',
            'instance_auto_stop': False, 'main_paper_run': False, 'final_test_executed': False}


def configs(mode):
    common = dict(noiseMultiplier=1., sensitivity=4096, maxDecisionSize=65536, minDecisionSize=0,
                  DPCreditorLoopInterval=50000, sendingLoopInterval=0, shaperCores=[2], workerCores=[3])
    return {
        'T2': dict(logLevel='WARNING', maxClients=20, queueSize=2097152, appName=f'gsmatched_{mode}_peer1',
                   shapedClient=dict(common, peer2Addr=PEERS[1], peer2Port=4567, sendingStrategy='BURST'),
                   unshapedServer=dict(bindAddr=PEERS[0], bindPort=8000, serverAddr='172.31.46.188:5555',
                                       checkQueuesInterval=500, cores=[4])),
        'T3': dict(logLevel='WARNING', maxPeers=1, maxStreamsPerPeer=20, queueSize=2097152,
                   appName=f'gsmatched_{mode}_peer2',
                   shapedServer=dict(common, serverCert='/tls/server.cert', serverKey='/tls/server.key',
                       listeningPort=4567, sensitivity=65536, maxDecisionSize=262144, strategy='BURST'),
                   unshapedClient=dict(checkQueuesInterval=500, cores=[4]))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--remote-root')
    p.add_argument('--baseline-root', default='/home/ubuntu/greenshaping-netshaper-20261020/runs/higgs_calibration_20261021_01')
    p.add_argument('--deadline-epoch', type=float)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    spec = plan()
    atomic_json(a.out / 'plan.json', spec)
    if a.plan_only:
        return
    if not a.remote_root or not a.deadline_epoch or not 120 <= a.deadline_epoch - time.time() <= 2700:
        p.error('staged remote root and a <=45 minute absolute window are required')
    r.require_available(list(r.INVENTORY['nodes']))
    root, base = a.remote_root, a.baseline_root
    if not root.startswith(r.INVENTORY['remote_root'] + '/runs/') or not re.fullmatch(r'[A-Za-z0-9_]+', Path(root).name):
        raise ValueError('invalid owned run root')
    ledger = SessionLedger(a.out / 'phase_ledger', spec)
    processes, containers = [], []
    cleanup_errors, nic_restore = [], {}
    phase_records = []

    def screen(label):
        def sample(node):
            raw = json.loads(r.python(node, PROBE.replace('SAMPLE_COUNT', '3').replace('INTERVAL', '1')))
            verdict = assess(raw)
            clock = r.python(node, "import subprocess;print(subprocess.check_output(['chronyc','tracking'],text=True))")
            offset = re.search(r'System time\s*:\s*([\d.]+)', clock)
            result = {'resource': raw, 'verdict': verdict, 'clock': clock}
            atomic_json(a.out / f'{label}_{node}_preflight.json', result)
            if not verdict['timing_candidate'] or not offset or float(offset[1]) > .01 or 'Normal' not in clock:
                raise RuntimeError('host resource/clock gate failed: ' + node)
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            list(pool.map(sample, r.INVENTORY['nodes']))

    def remaining(maximum):
        value = min(maximum, int(a.deadline_epoch - time.time() - 30))
        if value < 15:
            raise TimeoutError('approved cloud window ending; no extension')
        return value

    def start(node, argv, log, duration):
        out = r.python(node, f'''import json,subprocess,os
from pathlib import Path
root=Path({root!r})
with Path({log!r}).open('ab') as stream:
 p=subprocess.Popen(['timeout','--kill-after=5s',{str(duration)+'s'!r}]+{argv!r},
   stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True,
   env=dict(os.environ,PYTHONPATH=str(root/'app/src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
print(json.dumps({{'pid':p.pid}}))
''')
        processes.append((node, json.loads(out)['pid']))

    def stop_processes(items):
        while items:
            node, pid = items.pop()
            try:
                r.python(node, f'''import subprocess,time
from pathlib import Path
p=Path('/proc/{pid}/cmdline')
if p.exists() and {root.encode()!r} in p.read_bytes():
 subprocess.run(['sudo','-n','kill','-INT','--','-{pid}'],capture_output=True)
 time.sleep(.5)
 if p.exists() and {root.encode()!r} in p.read_bytes():
  subprocess.run(['sudo','-n','kill','-TERM','--','-{pid}'],capture_output=True)
''')
            except Exception as exc:
                cleanup_errors.append({'node': node, 'pid': pid, 'error': repr(exc)})

    def stop_containers():
        while containers:
            node, name = containers.pop()
            try:
                output = r.python(node, f'''import subprocess,json
name={name!r}
subprocess.run(['sudo','-n','docker','stop','--time','5',name],capture_output=True)
p=subprocess.run(['sudo','-n','docker','logs',name],capture_output=True,text=True)
v=subprocess.run(['sudo','-n','docker','inspect',name],capture_output=True,text=True)
subprocess.run(['sudo','-n','docker','rm',name],capture_output=True)
print(json.dumps({{'stdout':p.stdout,'stderr':p.stderr,'inspect':v.stdout}}))
''')
                atomic_json(a.out / (name + '.json'), json.loads(output))
            except Exception as exc:
                cleanup_errors.append({'node': node, 'container': name, 'error': repr(exc)})

    def collect_raw(phase, mode):
        errors = []
        for node in ('T2', 'T3'):
            for name in ['traffic.pcap', 'tcpdump.log', 'capture_durable.json']:
                try:
                    r.copy(node, a.out / f'{node}_{mode}_{name}', phase + '/' + name, get=True)
                except Exception as exc:
                    errors.append({'node': node, 'file': name, 'error': repr(exc)})
        try:
            r.python('T1', f'''import tarfile
from pathlib import Path
p=Path({phase!r})
with tarfile.open(p/'session_evidence.tar.gz','w:gz') as tar:
 for name in ['client','client.log','endpoints.json']:
  path=p/name
  if path.exists():tar.add(path,arcname=name)
''')
            r.copy('T1', a.out / f'{mode}_session_evidence.tar.gz', phase + '/session_evidence.tar.gz', get=True)
        except Exception as exc:
            errors.append({'node': 'T1', 'file': 'session_evidence', 'error': repr(exc)})
        atomic_json(a.out / (mode + '_collection.json'), {'errors': errors, 'all_collected': not errors})
        if errors:
            raise RuntimeError('raw evidence collection incomplete; remote originals retained')

    py = base + '/venv/bin/python'
    service_duration = remaining(2700)
    try:
        screen('initial')
        for node in r.INVENTORY['nodes']:
            r.python(node, f'''from pathlib import Path
root=Path({root!r});assert (root/'app/scripts/higgs_tls_endpoint.py').exists()
assert not (root/'measurement_started.json').exists()
(root/'measurement_started.json').write_text('true')
''')
        for node in ('T2', 'T3'):
            before = json.loads(r.python(node, '''import json,subprocess
from pathlib import Path
p=subprocess.run(['ethtool','-k','ens5'],capture_output=True,text=True,check=True)
print(json.dumps({'features':p.stdout,'mtu':int(Path('/sys/class/net/ens5/mtu').read_text())}))
'''))
            atomic_json(a.out / f'{node}_original_nic.json', before)
            restore = []
            feature_names = [('generic-receive-offload', 'gro'), ('generic-segmentation-offload', 'gso'),
                             ('tcp-segmentation-offload', 'tso')]
            for label, option in feature_names:
                match = re.search(r'^' + label + r': (on|off)', before['features'], re.M)
                if not match:
                    raise RuntimeError('NIC feature state unavailable: ' + label)
                restore += [option, match[1]]
            nic_restore[node] = restore
            r.python(node, "import subprocess\n"
                     "subprocess.run(['sudo','-n','ethtool','-K','ens5','gro','off','gso','off','tso','off'],check=True)")
        # Generate run-specific TLS credentials only on their destination hosts.
        token = json.loads(r.python('T4', f'''import secrets,subprocess,json,base64
from pathlib import Path
root=Path({root!r});tls=root/'tls';tls.mkdir()
subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','2','-subj','/CN=greenshaping-run',
'-addext','subjectAltName=IP:172.31.46.188,IP:172.31.34.237','-keyout',str(tls/'server.key'),'-out',str(tls/'server.cert')],capture_output=True,check=True)
(tls/'server.key').chmod(0o600);token=secrets.token_bytes(32);(root/'replica.token').write_bytes(token);(root/'replica.token').chmod(0o600)
print(json.dumps(base64.b64encode(token).decode()))
'''))
        r.python('T1', f'''import base64
from pathlib import Path
p=Path({root!r})/'replica.token';p.write_bytes(base64.b64decode({token!r}));p.chmod(0o600)
''')
        r.copy('T4', a.out / 'server.cert', root + '/tls/server.cert', get=True)
        r.copy('T1', a.out / 'server.cert', root + '/server.cert')
        r.python('T3', f'''import subprocess
from pathlib import Path
p=Path({root!r})/'tls';p.mkdir()
subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','2','-subj','/CN=netshaper-run',
 '-keyout',str(p/'server.key'),'-out',str(p/'server.cert')],capture_output=True,check=True);(p/'server.key').chmod(0o600)
''')
        start('T4', [py, root + '/app/scripts/evidence_replica_server.py', '--host', '172.31.46.188', '--port', '8000',
              '--root', root + '/replica_objects', '--cert', root + '/tls/server.cert', '--key', root + '/tls/server.key',
              '--token', root + '/replica.token', '--max-seconds', str(service_duration)], root + '/replica.log', service_duration)
        start('T4', [py, root + '/app/scripts/higgs_tls_endpoint.py', 'server', '--N', '20', '--store', base + '/data/store',
              '--out', root + '/server', '--host', '172.31.46.188', '--port', '5555', '--cert', root + '/tls/server.cert',
              '--key', root + '/tls/server.key', '--max-seconds', str(service_duration)], root + '/server.log', service_duration)
        r.python('T4', f'''import time
from pathlib import Path
p=Path({root!r})/'server/ready.json';deadline=time.monotonic()+30
while not p.exists():
 if time.monotonic()>deadline:raise TimeoutError('server not ready')
 time.sleep(.1)
''', timeout=40)
        for mode in spec['modes']:
            r.require_available(list(r.INVENTORY['nodes']))
            screen(mode)
            phase = root + '/' + mode
            ledger.start(mode, {'epoch': time.time()})
            for node in r.INVENTORY['nodes']:
                r.python(node, f'from pathlib import Path\nPath({phase!r}).mkdir()')
            owned_phase = []
            try:
                is_net = mode.startswith('netshaper')
                if is_net:
                    upstream_mode = 'no-shaping' if mode == 'netshaper_no_shaping' else 'shaping'
                    for node in ('T3', 'T2'):
                        peer = 'peer2' if node == 'T3' else 'peer1'
                        images = json.loads((r.EVIDENCE / f'{node}_runtime_provenance.json').read_text())['images']
                        image = next(i for i in images if i['tag'].endswith(peer + '-' + upstream_mode))
                        cfg = configs(mode)[node]
                        name = 'gsmatched-' + Path(root).name + '-' + mode + '-' + node.lower()
                        atomic_json(a.out / f'{node}_{mode}_config.json', cfg)
                        argv = ['sudo', '-n', 'docker', 'run', '-d', '--init', '--name', name, '--network', 'host',
                                '--shm-size=256m', '--memory=2g', '--pids-limit=256', '--mount',
                                f'type=bind,source={phase},target=/work,readonly']
                        if node == 'T3':
                            argv += ['--mount', f'type=bind,source={root}/tls,target=/tls,readonly']
                        argv += ['--entrypoint', '/usr/bin/timeout', image['id'], '--signal=INT', '--kill-after=5s',
                                 str(remaining(600)) + 's', '/root/peer_2' if node == 'T3' else '/root/peer_1', '/work/config.json']
                        containers.append((node, name))
                        r.python(node, f'''import json,subprocess
from pathlib import Path
(Path({phase!r})/'config.json').write_text(json.dumps({cfg!r}))
subprocess.run({argv!r},capture_output=True,check=True)
''')
                else:
                    for node, host, port, target in [('T3', PEERS[1], 5555, '172.31.46.188'), ('T2', PEERS[0], 8000, PEERS[1])]:
                        start(node, [py, root + '/app/scripts/tcp_evidence_relay.py', '--host', host, '--port', str(port),
                            '--target-host', target, '--target-port', '5555', '--ready', phase + '/relay_ready.json',
                            '--max-seconds', str(remaining(600))], phase + '/relay.log', remaining(600))
                        owned_phase.append(processes.pop())
                for node in ('T2', 'T3'):
                    start(node, ['sudo', '-n', 'tcpdump', '-U', '-n', '-i', 'ens5', '-s', '128', '-Z', 'ubuntu',
                        '-w', phase + '/traffic.pcap', BPF], phase + '/tcpdump.log', remaining(600))
                    owned_phase.append(processes.pop())
                    start(node, [py, root + '/app/scripts/fsync_capture.py', '--capture', phase + '/traffic.pcap',
                        '--checkpoint', phase + '/capture_durable.json', '--max-seconds', str(remaining(600))],
                        phase + '/fsync.log', remaining(600))
                    owned_phase.append(processes.pop())
                time.sleep(3)
                duration = remaining(600)
                result = r.python('T1', f'''import json,os,subprocess
from pathlib import Path
R=Path({root!r});phase=Path({phase!r});(phase/'endpoints.json').write_text(json.dumps([['172.31.34.237',8000]]*20))
argv=[{py!r},str(R/'app/scripts/higgs_tls_endpoint.py'),'client','--N','20','--bank',{base+'/train_n20.jsonl'!r},
'--reference',{base+'/reference_n20.json'!r},'--cert',str(R/'server.cert'),'--endpoints',str(phase/'endpoints.json'),
'--out',str(phase/'client'),'--modes',{('none' if is_net else mode)!r},'--repetitions','2','--max-seconds',{str(duration)!r},
'--replica-endpoint','https://172.31.46.188:8000','--replica-cert',str(R/'server.cert'),'--replica-token',str(R/'replica.token')]
with (phase/'client.log').open('ab') as log:
 p=subprocess.run(['timeout','--kill-after=5s',{str(duration)+'s'!r}]+argv,stdout=log,stderr=log,
 env=dict(os.environ,PYTHONPATH=str(R/'app/src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),start_new_session=True,timeout={duration+10})
print(json.dumps({{'returncode':p.returncode}}))
''', timeout=duration + 25)
                r.copy('T1', a.out / (mode + '_summary.json'), phase + '/client/summary.json', get=True)
                summary = json.loads((a.out / (mode + '_summary.json')).read_text())
                if json.loads(result)['returncode'] or not summary['all_passed'] or summary['replica_acknowledged_sessions'] != 24:
                    raise RuntimeError('invalid client/replica phase; no automatic replay: ' + mode)
            finally:
                stop_processes(owned_phase)
                stop_containers()
                collect_raw(phase, mode)
            if cleanup_errors:
                raise RuntimeError('phase cleanup uncertain; no further acquisition')
            captures = {}
            for node in ('T2', 'T3'):
                log = (a.out / f'{node}_{mode}_tcpdump.log').read_text()
                drop = re.search(r'(\d+) packets dropped by kernel', log)
                mtu = json.loads((a.out / f'{node}_original_nic.json').read_text())['mtu']
                parsed = audit(a.out / f'{node}_{mode}_traffic.pcap', PEERS, mtu)
                if (not drop or int(drop[1]) or not parsed['wire_size_screen_passed']
                        or set(parsed['direction_counts']) != {'up', 'down'}):
                    raise RuntimeError('capture validity gate failed: ' + node + '/' + mode)
                captures[node] = parsed
            record = {'passed': True, 'sessions': 24, 'captures': captures, 'epoch': time.time()}
            ledger.commit(mode, record)
            phase_records.append(record)
            atomic_json(a.out / 'progress.json', ledger.checkpoint())
        atomic_json(a.out / 'summary.json', {'all_passed': True, 'sessions': 168, 'phase_audit': ledger.checkpoint(),
            'scope': 'TRAIN matched observer calibration only', 'main_paper_run': False, 'energy_measured': False})
    except Exception as exc:
        atomic_json(a.out / 'failure.json', {'error': repr(exc), 'epoch': time.time(), 'auto_retry': False})
        raise
    finally:
        stop_containers()
        stop_processes(processes)
        for node, features in nic_restore.items():
            try:
                r.python(node, f"import subprocess;subprocess.run({['sudo','-n','ethtool','-K','ens5',*features]!r},check=True)")
            except Exception as exc:
                cleanup_errors.append({'node': node, 'nic_restore_error': repr(exc)})
        atomic_json(a.out / 'cleanup.json', {'errors': cleanup_errors, 'instance_auto_stop': False})
        atomic_json(a.out / 'manifest.json', {p.name: file_sha256(p) for p in a.out.iterdir() if p.is_file() and p.name != 'manifest.json'})
        if os.uname().sysname == 'Darwin':
            notice = 'display notification "EC2 batch ended. Review results and stop instances." with title "Green Shaping"'
            subprocess.run(['/usr/bin/osascript', '-e', notice],
                           capture_output=True, timeout=10)


if __name__ == '__main__':
    main()
