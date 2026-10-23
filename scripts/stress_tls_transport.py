"""Repeated tiny DP sessions on two persistent links; engineering stress test only."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from greenshaping.shaping.pipeline import DPParams, ShapingConfig
from greenshaping.testbed.queries import run_query
from greenshaping.testbed.transport import ShapedTransport, TLSToolServer
from greenshaping.utils.io import write_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--sessions', type=int, default=150)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cert, key = out / 'cert.pem', out / 'key.pem'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1', '-subj', '/CN=127.0.0.1',
                    '-addext', 'subjectAltName=IP:127.0.0.1', '-keyout', str(key), '-out', str(cert)],
                   check=True, capture_output=True)
    key.chmod(0o600)
    table = pd.DataFrame({'x': range(20)})
    servers = [TLSToolServer('127.0.0.1', 0, str(cert), str(key), lambda f, p: run_query(table, f, p)) for _ in range(2)]
    cfg = ShapingConfig(dp=DPParams(mode='arrivals', interval=.01, horizon=.04, delta=4096,
                                    volume_cap_bytes=2048, bias=30000))
    tr = ShapedTransport([s.address for s in servers], str(cert), cfg, start_lead_s=.1)
    status = {'state': 'running', 'completed': 0, 'sessions': args.sessions}
    try:
        for i in range(args.sessions):
            tr.begin()
            assert tr.call_round('record_count', [{'site': i % 2, 'params': {'limit_rows': 8}}])[0]['matching_rows'] == 8
            tr.finish()
            assert all(not c.error and not c.pending and len(c.logs) == 4 for c in tr.channels)
            assert all(len(s.channels) == 1 and not s.channels[0].error and not s.channels[0].pending for s in servers)
            status['completed'] = i + 1
            write_json(status, out / 'status.json')
            if (i + 1) % 10 == 0:
                print(status, flush=True)
        status['state'] = 'complete'
    except Exception as e:
        status.update(state='failed', error=str(e))
        raise
    finally:
        write_json(status, out / 'status.json')
        tr.close()
        for s in servers:
            s.stop()


if __name__ == '__main__':
    main()
