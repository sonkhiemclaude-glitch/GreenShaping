"""Verify actual response bytes and record latency; no privacy or energy inference."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import random
import time
from pathlib import Path

from ec2_payload_server import SIZES, payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--mode', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--repetitions', type=int, default=3)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    order = [(size, rep) for rep in range(a.repetitions) for size in SIZES]
    random.Random(20261020).shuffle(order)
    rows = []
    for size, rep in order:
        start = time.time()
        row: dict = {'mode': a.mode, 'size': size, 'repetition': rep, 'start_epoch': start}
        conn = http.client.HTTPConnection(a.host, a.port, timeout=15)
        try:
            conn.request('GET', f'/payload/{size}?trial={a.mode}-{rep}-{size}', headers={'Connection': 'close'})
            response = conn.getresponse()
            body = response.read()
            sha = hashlib.sha256(body).hexdigest()
            row.update(status=response.status, received_bytes=len(body), sha256=sha,
                       correct=response.status == 200 and len(body) == size and
                       body == payload(size) and sha == response.getheader('X-Payload-SHA256'))
            if not row['correct']:
                raise RuntimeError('response content/length/hash mismatch')
        except Exception as e:
            row.update(correct=False, error=str(e))
            raise
        finally:
            row['completion_latency_s'] = time.time() - start
            rows.append(row)
            out.write_text(json.dumps(rows, indent=2) + '\n')
            conn.close()
    print(json.dumps({'mode': a.mode, 'requests': len(rows), 'correct': sum(r['correct'] for r in rows)}), flush=True)


if __name__ == '__main__':
    main()
