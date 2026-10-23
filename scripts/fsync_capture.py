"""Flush a bounded packet-buffered capture and record its last durable offset."""
import argparse
import os
import time
from pathlib import Path

from greenshaping.utils.evidence import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--max-seconds', type=int, required=True)
    a = p.parse_args()
    if not 1 <= a.max_seconds <= 5400:
        p.error('finite lifetime required')
    deadline = time.monotonic() + a.max_seconds
    while time.monotonic() < deadline:
        if a.capture.exists():
            with a.capture.open('rb') as f:
                os.fsync(f.fileno())
                offset = os.fstat(f.fileno()).st_size
            atomic_json(a.checkpoint, {'durable_bytes': offset, 'epoch': time.time(),
                                      'capture_complete': False, 'may_have_incomplete_last_packet': True})
        time.sleep(1)


if __name__ == '__main__':
    main()
