"""Bounded throughput tests using private addresses on the four existing EC2 VMs."""

import argparse
import concurrent.futures
import hashlib
import json
import shlex
import time

import ec2_runner as r

NAME = "capacity_01"
LOCAL = r.EVIDENCE / NAME
REMOTE = r.INVENTORY["remote_root"] + "/runs/" + NAME
NODES = r.INVENTORY["nodes"]
CONTAINER_MEMORY = "4g"
CONTAINERS = []
SERVERS = []
ROWS = []


def parallel(fn, items):
    with concurrent.futures.ThreadPoolExecutor(4) as p:
        return list(p.map(fn, items))


def save(name, obj):
    (LOCAL / name).write_text(json.dumps(obj, indent=2) + "\n")


def status(phase, **kw):
    obj = dict(phase=phase, completed_trials=len(ROWS), **kw)
    save("status.json", obj)
    print(json.dumps(obj), flush=True)


def metrics(node):
    return node, json.loads(
        r.python(
            node,
            """import json,subprocess,time
from pathlib import Path
p=subprocess.run(['sudo','-n','/usr/sbin/ethtool','-S','ens5'],capture_output=True,text=True)
print(json.dumps({'time_ns':time.time_ns(),'proc_stat':Path('/proc/stat').read_text(),'net_dev':Path('/proc/net/dev').read_text(),'ena':p.stdout,'ena_error':p.stderr,'snmp':Path('/proc/net/snmp').read_text()}))
""",
        )
    )


def server(node, port, label):
    pid = int(
        r.python(
            node,
            f"""import subprocess
from pathlib import Path
with (Path({REMOTE!r})/{label + "_server.log"!r}).open('ab') as log:
 p=subprocess.Popen(['timeout','--signal=TERM','--kill-after=3s','1500s','iperf3','-s','-B',{NODES[node]["private_ip"]!r},'-p',{str(port)!r}],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(p.pid)
""",
        )
    )
    SERVERS.append((node, pid, label))
    time.sleep(0.5)


def trial(label, client, target, port, streams=4, offered=None, reverse=False, seconds=8, repetition=1):
    status(label, state="running")
    before = dict(parallel(metrics, NODES))
    args = [
        "iperf3",
        "-c",
        NODES[target]["private_ip"],
        "-p",
        str(port),
        "-P",
        str(streams),
        "-t",
        str(seconds),
        "-O",
        "2",
        "-J",
        "--connect-timeout",
        "5000",
    ]
    if offered is not None:
        args += ["-b", str(int(offered / streams))]  # -b is PER STREAM.
    if reverse:
        args += ["-R"]
    result = r.run(client, shlex.join(["timeout", "--kill-after=3s", str(seconds + 30) + "s", *args]), timeout=seconds + 45)
    after = dict(parallel(metrics, NODES))
    key = f"{len(ROWS):03d}_{label}_r{repetition}"
    try:
        data = json.loads(result.stdout)
    except ValueError:
        data = {"error": "invalid JSON", "stdout": result.stdout}
    save(
        key + ".json",
        dict(
            client=client,
            target=target,
            args=args,
            returncode=result.returncode,
            stderr=result.stderr,
            iperf=data,
            before=before,
            after=after,
        ),
    )
    end = data.get("end", {})
    row = dict(
        id=key,
        label=label,
        streams=streams,
        reverse=reverse,
        offered_aggregate_bps=offered,
        repetition=repetition,
        seconds=seconds,
        receiver_bps=end.get("sum_received", {}).get("bits_per_second"),
        sender_bps=end.get("sum_sent", {}).get("bits_per_second"),
        retransmits=end.get("sum_sent", {}).get("retransmits"),
        error=data.get("error"),
        returncode=result.returncode,
    )
    row["reached_10Gbps"] = bool(row["receiver_bps"] and row["receiver_bps"] >= 10_000_000_000)
    ROWS.append(row)
    save("summary.json", ROWS)
    print(json.dumps(row), flush=True)
    return row


def stop_servers():
    while SERVERS:
        node, pid, label = SERVERS.pop()
        r.python(
            node,
            f"""import os,signal
from pathlib import Path
p=Path('/proc/{pid}/cmdline')
if p.exists() and b'iperf3' in p.read_bytes():
 try:os.killpg({pid},signal.SIGTERM)
 except ProcessLookupError:pass
""",
        )
        r.copy(node, LOCAL / (label + "_server.log"), REMOTE + "/" + label + "_server.log", get=True)


def stop_containers():
    while CONTAINERS:
        node, name = CONTAINERS.pop()
        out = r.python(
            node,
            f"""import subprocess,json
subprocess.run(['sudo','-n','docker','stop','--time','3',{name!r}],capture_output=True)
a=subprocess.run(['sudo','-n','docker','logs','--tail','150',{name!r}],capture_output=True,text=True)
b=subprocess.run(['sudo','-n','docker','inspect',{name!r}],capture_output=True,text=True)
subprocess.run(['sudo','-n','docker','rm',{name!r}],capture_output=True)
print(json.dumps({{'stdout':a.stdout,'stderr':a.stderr,'inspect':json.loads(b.stdout) if b.returncode==0 else None}}))
""",
        )
        (LOCAL / (name + "_runtime.json")).write_text(out)


def proxies(mode, suffix=""):
    phase = mode + suffix
    common = dict(
        noiseMultiplier=1.0,
        sensitivity=65536,
        maxDecisionSize=16777216,
        minDecisionSize=0,
        DPCreditorLoopInterval=1000,
        sendingLoopInterval=0,
        shaperCores=[2],
        workerCores=[1, 3, 5, 7],
    )
    configs = {
        "T2": dict(
            logLevel="WARNING",
            maxClients=16,
            queueSize=8388608,
            appName=f"gs_{NAME}_{phase}_p1",
            shapedClient=dict(common, peer2Addr=NODES["T3"]["private_ip"], peer2Port=4567, sendingStrategy="BURST"),
            unshapedServer=dict(
                bindAddr=NODES["T2"]["private_ip"],
                bindPort=8000,
                serverAddr=NODES["T4"]["private_ip"] + ":5555",
                checkQueuesInterval=50,
                cores=[0, 4],
            ),
        ),
        "T3": dict(
            logLevel="WARNING",
            maxPeers=1,
            maxStreamsPerPeer=16,
            queueSize=8388608,
            appName=f"gs_{NAME}_{phase}_p2",
            shapedServer=dict(common, serverCert="/tls/server.cert", serverKey="/tls/server.key", listeningPort=4567, strategy="BURST"),
            unshapedClient=dict(checkQueuesInterval=50, cores=[0, 4]),
        ),
    }
    for node in ("T3", "T2"):
        cfg = configs[node]
        save(f"{node}_{phase}_config.json", cfg)
        images = json.loads((r.EVIDENCE / f"{node}_runtime_provenance.json").read_text())["images"]
        peer = "peer2" if node == "T3" else "peer1"
        image = next(v for v in images if v["tag"].endswith(peer + "-" + mode.replace("_", "-")))
        name = f"gsnet-{NAME}-{phase}-{node.lower()}"
        args = [
            "sudo",
            "-n",
            "docker",
            "run",
            "-d",
            "--init",
            "--name",
            name,
            "--network",
            "host",
            "--shm-size=1g",
            f"--memory={CONTAINER_MEMORY}",
            "--pids-limit=512",
            "--mount",
            f"type=bind,source={REMOTE}/{phase},target=/work,readonly",
        ]
        if node == "T3":
            args += ["--mount", f"type=bind,source={REMOTE}/tls,target=/tls,readonly"]
        args += [
            "--entrypoint",
            "/usr/bin/timeout",
            image["id"],
            "--signal=INT",
            "--kill-after=5s",
            "900s",
            "/root/peer_2" if node == "T3" else "/root/peer_1",
            "/work/config.json",
        ]
        r.python(
            node,
            f"""import subprocess,json
from pathlib import Path
p=Path({REMOTE!r})/{phase!r};p.mkdir()
(p/'config.json').write_text(json.dumps({cfg!r},indent=2))
a=subprocess.run({args!r},capture_output=True,text=True)
assert a.returncode==0,a.stderr
print(a.stdout)
""",
        )
        CONTAINERS.append((node, name))
    time.sleep(4)
    for node, name in CONTAINERS:
        r.python(
            node,
            f"""import json,subprocess
v=json.loads(subprocess.check_output(['sudo','-n','docker','inspect',{name!r}],text=True))[0]
assert v['State']['Running'],v['State']
""",
        )


def main():
    global NAME, LOCAL, REMOTE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    r.require_available(list(NODES))
    if not args.run_name.replace("_", "").isalnum():
        raise ValueError("run-name must use letters, digits and underscores")
    NAME = args.run_name
    LOCAL = r.EVIDENCE / NAME
    REMOTE = r.INVENTORY["remote_root"] + "/runs/" + NAME
    LOCAL.mkdir(exist_ok=False)
    save(
        "protocol.json",
        dict(
            run=NAME,
            nodes=NODES,
            target_bps=10_000_000_000,
            streams=4,
            iperf_omit_seconds=2,
            measured="TCP application receiver goodput",
            source_commit=r.INVENTORY["source_commit"],
            runtime="Pinned upstream binary images, not rebuilt from recorded source",
            shaping="1 ms interval; sigma 65536 bytes; max decision 16 MiB; performance configuration only",
            energy_measured=False,
            privacy_certified=False,
            limitations=[
                "Short burst assessment, not sustained bandwidth certification",
                "Single QUIC connection between middleboxes",
                "No packet capture during throughput tests",
                "Physical host independence unverified",
            ],
        ),
    )
    try:
        parallel(lambda n: r.python(n, f"from pathlib import Path\nPath({REMOTE!r}).mkdir(parents=True,exist_ok=False)"), NODES)
        server("T3", 5201, "raw_t3")
        for streams in (1, 4):
            trial(f"raw_T2_T3_P{streams}", "T2", "T3", 5201, streams=streams)
        trial("raw_T2_T3_P4_reverse", "T2", "T3", 5201, reverse=True)
        stop_servers()
        server("T2", 5201, "raw_t2")
        trial("raw_T1_T2_P4", "T1", "T2", 5201)
        stop_servers()
        server("T4", 5201, "raw_t4")
        trial("raw_T3_T4_P4", "T3", "T4", 5201)
        for rep in (1, 2, 3):
            trial("raw_T1_T4_P4_10G", "T1", "T4", 5201, offered=10_000_000_000, repetition=rep)
        stop_servers()
        r.python(
            "T3",
            f"""from pathlib import Path
import subprocess
p=Path({REMOTE!r})/'tls';p.mkdir()
subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','2','-subj','/CN=netshaper-capacity','-keyout',str(p/'server.key'),'-out',str(p/'server.cert')],capture_output=True,check=True)
(p/'server.key').chmod(0o600)
""",
        )
        server("T4", 5555, "netshaper_app")
        for mode in ("no_shaping", "shaping"):
            proxies(mode)
            probe = trial(mode + "_compatibility", "T1", "T2", 8000, streams=1, offered=10_000_000, seconds=3)
            if probe["error"] or probe["returncode"] or not probe["receiver_bps"]:
                status(mode + "_incompatible", state="running")
                stop_containers()
                continue
            for reverse in (False, True):
                direction = "reverse" if reverse else "forward"
                for offered in (500_000_000, 1_000_000_000, 3_000_000_000):
                    trial(f"{mode}_{direction}_{offered}", "T1", "T2", 8000, offered=offered, reverse=reverse)
                for rep in (1, 2, 3):
                    trial(f"{mode}_{direction}_10G", "T1", "T2", 8000, offered=10_000_000_000, reverse=reverse, seconds=10, repetition=rep)
            stop_containers()
        status("complete", state="complete")
    except Exception as exc:
        status("failed", state="failed", error=str(exc))
        raise
    finally:
        stop_containers()
        stop_servers()
        save(
            "final_runtime.json",
            dict(
                parallel(
                    lambda n: (
                        n,
                        r.python(
                            n,
                            """import subprocess,json
print(json.dumps({
 'containers':subprocess.check_output(['sudo','-n','docker','ps','--format','{{.Names}}'],text=True),
 'listeners':subprocess.check_output(['ss','-lntup'],text=True)}))
""",
                        ),
                    ),
                    NODES,
                )
            ),
        )
        save(
            "manifest.json",
            {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in LOCAL.iterdir() if p.is_file() and p.name != "manifest.json"},
        )


if __name__ == "__main__":
    main()
