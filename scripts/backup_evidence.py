"""Create/verify/restore explicit evidence sources; credentials are excluded."""
import argparse
import json
from pathlib import Path

from greenshaping.utils.evidence_backup import snapshot, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=['snapshot', 'verify', 'restore'])
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--source', action='append', default=[], help='label=/absolute/evidence/path')
    p.add_argument('--restore-to', type=Path)
    a = p.parse_args()
    if a.operation == 'snapshot':
        result = snapshot({k: Path(v) for k, v in (s.split('=', 1) for s in a.source)}, a.destination, a.name)
    else:
        if a.operation == 'restore' and a.restore_to is None:
            p.error('restore requires --restore-to')
        result = verify(a.destination, a.name, a.restore_to if a.operation == 'restore' else None)
    print(json.dumps(result, indent=2))
    if not result['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
