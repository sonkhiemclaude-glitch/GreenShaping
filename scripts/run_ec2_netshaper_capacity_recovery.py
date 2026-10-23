"""Controlled follow-up: 16 GiB cap, fresh proxies per trial, same four instances."""

import argparse
import json
import shutil

import run_ec2_netshaper_capacity as b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", default="capacity_recovery_01")
    args = parser.parse_args()
    b.r.require_available(list(b.NODES))
    if not args.run_name.replace("_", "").isalnum():
        raise ValueError("run-name must use letters, digits and underscores")
    b.NAME = args.run_name
    b.LOCAL = b.r.EVIDENCE / b.NAME
    b.REMOTE = b.r.INVENTORY["remote_root"] + "/runs/" + b.NAME
    b.CONTAINER_MEMORY = "16g"
    b.LOCAL.mkdir(exist_ok=args.resume)
    if args.resume:
        b.ROWS.extend(json.loads((b.LOCAL / "summary.json").read_text()))
    b.save(
        "resume_protocol.json" if args.resume else "protocol.json",
        dict(
            run=b.NAME,
            target_bps=10_000_000_000,
            streams=4,
            measured="TCP application receiver goodput",
            nodes=b.NODES,
            rationale="capacity_01 imposed 4 GiB and T2 was OOM-killed. Increase cap to 16 GiB; fresh middleboxes each trial.",
            container_memory="16 GiB per middlebox",
            duration_seconds=5,
            omit_seconds=2,
            matrix=dict(
                modes=["no_shaping", "shaping"],
                directions=["forward", "reverse"],
                offered_aggregate_bps=[3_000_000_000, 10_000_000_000],
                reps_3G=1,
                reps_10G=3,
            ),
            runtime="Same immutable upstream image IDs as functional_01; not rebuilt from reference commit",
            shaping="Same 1ms/sigma65536/maxDecision16MiB performance parameters as capacity_01",
            energy_measured=False,
            privacy_certified=False,
            limitations=[
                "Very short burst tests, not sustained 10 Gbps certification",
                "No matched privacy comparison",
                "Separate retry protocol; original failures retained",
                "No full payload-integrity validation by iperf3",
            ],
        ),
    )
    try:
        b.parallel(
            lambda n: b.r.python(n, f"from pathlib import Path\nPath({b.REMOTE!r}).mkdir(parents=True,exist_ok={args.resume!r})"), b.NODES
        )
        if not args.resume:
            b.r.python(
                "T3",
                f"""from pathlib import Path
import subprocess
p=Path({b.REMOTE!r})/'tls';p.mkdir()
subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','2','-subj','/CN=netshaper-capacity-recovery','-keyout',str(p/'server.key'),'-out',str(p/'server.cert')],capture_output=True,check=True)
(p/'server.key').chmod(0o600)
""",
            )
        b.server("T4", 5555, "app_resume" if args.resume else "netshaper_app")
        for mode in ("no_shaping", "shaping"):
            for reverse in (False, True):
                direction = "reverse" if reverse else "forward"
                for offered, reps in ((3_000_000_000, 1), (10_000_000_000, 3)):
                    for rep in range(1, reps + 1):
                        label = f"{mode}_{direction}_{'10G' if offered == 10_000_000_000 else offered}"
                        if any(x["label"] == label and x["repetition"] == rep for x in b.ROWS):
                            continue
                        suffix = f"_{direction}_{offered}_{rep}"
                        b.proxies(mode, suffix)
                        row = b.trial(
                            f"{mode}_{direction}_{'10G' if offered == 10_000_000_000 else offered}",
                            "T1",
                            "T2",
                            8000,
                            offered=offered,
                            reverse=reverse,
                            seconds=5,
                            repetition=rep,
                        )
                        b.stop_containers()
                        if row["error"] or row["returncode"]:
                            # A failed proxied control session can leave iperf's server busy.
                            b.stop_servers()
                            b.server("T4", 5555, f"app_after_{len(b.ROWS)}")
        b.status("complete", state="complete")
    except BaseException as exc:
        b.status("failed", state="failed", error=repr(exc))
        raise
    finally:
        b.stop_containers()
        b.stop_servers()
        b.save(
            "final_runtime.json",
            dict(
                b.parallel(
                    lambda n: (
                        n,
                        json.loads(
                            b.r.python(
                                n,
                                """import subprocess,json
print(json.dumps({'containers':subprocess.check_output(['sudo','-n','docker','ps','--format','{{.Names}}'],text=True),
 'listeners':subprocess.check_output(['ss','-lntup'],text=True)}))
""",
                            )
                        ),
                    ),
                    b.NODES,
                )
            ),
        )
        for name in ("run_ec2_netshaper_capacity.py", "run_ec2_netshaper_capacity_recovery.py"):
            shutil.copyfile(__file__.replace("run_ec2_netshaper_capacity_recovery.py", name), b.LOCAL / name)


if __name__ == "__main__":
    main()
