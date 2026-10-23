"""Bounded TRAIN HIGGS correctness through pinned upstream NetShaper images.

Requires the existing N=20 TLS endpoint. No energy/privacy/throughput claim.
Every remote measurement process and container has an independent timeout.
"""
import json
import time

import ec2_runner as r


def main():
    r.require_available(['T1', 'T2', 'T3', 'T4'])
    direct = json.loads((r.ROOT / 'outputs/ec2_higgs_20261021/tls_n20_summary.json').read_text())
    if not direct.get('all_passed') or direct.get('sessions') != 60:
        raise RuntimeError('Direct TLS calibration must complete before proxy acquisition')
    root = r.INVENTORY['remote_root'] + '/runs/higgs_calibration_20261021_01'
    out = r.ROOT / 'outputs/ec2_higgs_20261021/netshaper_higgs_01'
    out.mkdir(exist_ok=False)
    owned = []
    try:
        r.python('T3', f'''from pathlib import Path
import subprocess
p=Path({root!r})/'netshaper_tls';p.mkdir()
subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','2','-subj','/CN=netshaper-functional','-keyout',str(p/'server.key'),'-out',str(p/'server.cert')],check=True,capture_output=True)
(p/'server.key').chmod(0o600)
''')
        for mode in ('no-shaping', 'shaping'):
            common = dict(noiseMultiplier=1., sensitivity=4096, maxDecisionSize=65536,
                          minDecisionSize=0, DPCreditorLoopInterval=50000,
                          sendingLoopInterval=0, shaperCores=[2], workerCores=[3])
            cfgs = {
                'T2': dict(logLevel='WARNING', maxClients=20, queueSize=2097152,
                           appName=f'higgs01_{mode}_peer1',
                           shapedClient=dict(common, peer2Addr='172.31.38.41', peer2Port=4567, sendingStrategy='BURST'),
                           unshapedServer=dict(bindAddr='172.31.34.237', bindPort=8000,
                                               serverAddr='172.31.46.188:5555', checkQueuesInterval=500, cores=[4])),
                'T3': dict(logLevel='WARNING', maxPeers=1, maxStreamsPerPeer=20, queueSize=2097152,
                           appName=f'higgs01_{mode}_peer2',
                           shapedServer=dict(common, serverCert='/tls/server.cert', serverKey='/tls/server.key',
                                             listeningPort=4567, sensitivity=65536, maxDecisionSize=262144, strategy='BURST'),
                           unshapedClient=dict(checkQueuesInterval=500, cores=[4]))}
            for node in ('T3', 'T2'):
                peer = 'peer2' if node == 'T3' else 'peer1'
                images = json.loads((r.EVIDENCE / f'{node}_runtime_provenance.json').read_text())['images']
                image = next(i for i in images if i['tag'].endswith(peer + '-' + mode))
                name = f'gs-higgs01-{mode}-{node.lower()}'
                folder = root + '/netshaper_' + mode
                cfg = cfgs[node]
                (out / f'{node}_{mode}_config.json').write_text(json.dumps(cfg, indent=2))
                cmd = ['sudo', '-n', 'docker', 'run', '-d', '--init', '--name', name, '--network', 'host',
                       '--shm-size=256m', '--memory=2g', '--pids-limit=256', '--mount',
                       f'type=bind,source={folder},target=/work,readonly']
                if node == 'T3':
                    cmd += ['--mount', f'type=bind,source={root}/netshaper_tls,target=/tls,readonly']
                cmd += ['--entrypoint', '/usr/bin/timeout', image['id'], '--signal=INT', '--kill-after=5s',
                        '180s', '/root/peer_2' if node == 'T3' else '/root/peer_1', '/work/config.json']
                owned.append((node, name, mode))
                r.python(node, f'''from pathlib import Path
import json,subprocess
p=Path({folder!r});p.mkdir();(p/'config.json').write_text(json.dumps({cfg!r}))
print(subprocess.check_output({cmd!r},text=True))
''')
            time.sleep(3)
            result = r.python('T1', f'''from pathlib import Path
import json,os,subprocess
R=Path({root!r});(R/'endpoints_netshaper_n20.json').write_text(json.dumps([['172.31.34.237',8000]]*20))
argv=[str(R/'venv/bin/python'),str(R/'app/scripts/higgs_tls_endpoint.py'),'client','--N','20','--bank',str(R/'train_n20.jsonl'),'--out',str(R/{('client_netshaper_'+mode)!r}),'--reference',str(R/'reference_n20.json'),'--cert',str(R/'server.cert'),'--endpoints',str(R/'endpoints_netshaper_n20.json'),'--repetitions','1','--modes','none','--max-seconds','90']
p=subprocess.run(['timeout','--kill-after=5s','110s']+argv,env=dict(os.environ,PYTHONPATH=str(R/'app/src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),capture_output=True,text=True,timeout=120,start_new_session=True)
print(json.dumps({{'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}}))
''', timeout=130)
            (out / f'{mode}_client_process.json').write_text(result)
            r.copy('T1', out / f'{mode}_summary.json', root + '/client_netshaper_' + mode + '/summary.json', get=True)
            cleanup(owned, out)
            if json.loads(result)['returncode']:
                raise RuntimeError('Retained NetShaper functional findings; no automatic replay')
        (out / 'scope.json').write_text(json.dumps(dict(scope='TRAIN HIGGS, 20 logical sites, four physical EC2 roles',
            privacy_certified=False, physical_energy_measured=False, throughput_benchmark=False,
            shaping_budget_matched=False, final_test_executed=False), indent=2))
    finally:
        cleanup(owned, out)


def cleanup(owned, out):
    while owned:
        node, name, mode = owned.pop()
        result = r.python(node, f'''import subprocess,json
name={name!r}
subprocess.run(['sudo','-n','docker','stop','--time','5',name],capture_output=True)
p=subprocess.run(['sudo','-n','docker','logs',name],capture_output=True,text=True)
i=subprocess.run(['sudo','-n','docker','inspect',name],capture_output=True,text=True)
subprocess.run(['sudo','-n','docker','rm',name],capture_output=True)
print(json.dumps({{'stdout':p.stdout,'stderr':p.stderr,'inspect':i.stdout}}))
''')
        (out / f'{node}_{mode}_runtime.json').write_text(result)


if __name__ == '__main__':
    main()
