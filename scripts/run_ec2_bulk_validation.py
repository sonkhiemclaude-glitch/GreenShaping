"""Independent fixed-length, CRC-checked throughput validation after the iperf campaign."""

import json
import shlex
import shutil
from pathlib import Path

import run_ec2_netshaper_capacity as b


def main():
    b.r.require_available(list(b.NODES))
    b.NAME = "bulk_verify_01"
    b.LOCAL = b.r.EVIDENCE / b.NAME
    b.REMOTE = b.r.INVENTORY["remote_root"] + "/runs/" + b.NAME
    b.CONTAINER_MEMORY = "16g"
    b.LOCAL.mkdir(exist_ok=False)
    b.save(
        "protocol.json",
        dict(
            target_aggregate_bps=10_000_000_000,
            streams=4,
            bytes_per_stream=1073741824,
            receiver_metric="4 GiB / receiver-local (last byte - first byte) across all 4 streams",
            modes=["direct_forward", "no_shaping_forward", "no_shaping_reverse", "shaping_forward", "shaping_reverse"],
            repetitions=1,
            crc32_validated=True,
            source="same pinned upstream binaries",
            rationale="Cross-check iperf3 control/data-stream interaction; fixed data framing, no shared control stream",
            limitations=[
                "Single short transfer per condition",
                "Synthetic zero payload with CRC32 integrity check",
                "Not a privacy/energy comparison",
            ],
        ),
    )
    pid = None
    rows = []
    try:
        b.parallel(lambda n: b.r.python(n, f"from pathlib import Path\nPath({b.REMOTE!r}).mkdir(parents=True,exist_ok=False)"), b.NODES)
        for node in ("T1", "T4"):
            b.r.copy(node, Path(__file__).parent / "ec2_bulk_probe.py", b.r.INVENTORY["remote_root"] + "/ec2_bulk_probe.py")
        b.r.python(
            "T3",
            f"""import shutil
from pathlib import Path
p=Path({b.REMOTE!r})/'tls';p.mkdir()
old=Path({b.r.INVENTORY["remote_root"] + "/runs/capacity_recovery_01/tls"!r})
for name in ('server.key','server.cert'):shutil.copyfile(old/name,p/name)
(p/'server.key').chmod(0o600)
""",
        )
        pid = int(
            b.r.python(
                "T4",
                f"""import subprocess
from pathlib import Path
with (Path({b.REMOTE!r})/'server.jsonl').open('ab') as log:
 args=['timeout','--kill-after=3s','330s','python3','-u',
       {b.r.INVENTORY["remote_root"] + "/ec2_bulk_probe.py"!r},'--server','--host',{b.NODES["T4"]["private_ip"]!r}]
 p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(p.pid)
""",
            )
        )
        for mode, reverse in (("direct", False), ("no_shaping", False), ("no_shaping", True), ("shaping", False), ("shaping", True)):
            label = mode + ("_reverse" if reverse else "_forward")
            b.status(label, state="running")
            if mode != "direct":
                b.proxies(mode, "_bulk_" + ("reverse" if reverse else "forward"))
            before = dict(b.parallel(b.metrics, b.NODES))
            host = b.NODES["T4" if mode == "direct" else "T2"]["private_ip"]
            port = 5555 if mode == "direct" else 8000
            args = [
                "timeout",
                "--kill-after=3s",
                "85s",
                "python3",
                b.r.INVENTORY["remote_root"] + "/ec2_bulk_probe.py",
                "--host",
                host,
                "--port",
                str(port),
                "--mode",
                "GET" if reverse else "PUT",
                "--trial",
                label,
            ]
            result = b.r.run("T1", shlex.join(args), timeout=95)
            after = dict(b.parallel(b.metrics, b.NODES))
            try:
                data = json.loads(result.stdout)
            except ValueError:
                data = {"error": "No valid bulk JSON", "stdout": result.stdout}
            b.save(
                label + ".json",
                dict(args=args, returncode=result.returncode, stderr=result.stderr, result=data, before=before, after=after),
            )
            row = dict(
                label=label,
                returncode=result.returncode,
                receiver_bps=data.get("receiver_bps"),
                crc_valid=data.get("crc_valid"),
                total_bytes=data.get("total_bytes"),
                error=data.get("error"),
            )
            rows.append(row)
            b.save("summary.json", rows)
            print(json.dumps(row), flush=True)
            b.stop_containers()
        b.status("complete", state="complete", bulk_trials=len(rows))
    except BaseException as exc:
        b.status("failed", state="failed", error=repr(exc))
        raise
    finally:
        b.stop_containers()
        if pid:
            b.r.python(
                "T4",
                f"""import os,signal
from pathlib import Path
p=Path('/proc/{pid}/cmdline')
if p.exists() and b'ec2_bulk_probe.py' in p.read_bytes():
 try:os.killpg({pid},signal.SIGTERM)
 except ProcessLookupError:pass
""",
            )
            b.r.copy("T4", b.LOCAL / "server.jsonl", b.REMOTE + "/server.jsonl", get=True)
        for name in ("ec2_bulk_probe.py", "run_ec2_bulk_validation.py", "run_ec2_netshaper_capacity.py"):
            shutil.copyfile(Path(__file__).parent / name, b.LOCAL / name)


if __name__ == "__main__":
    main()
