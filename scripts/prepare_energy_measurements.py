"""Convert paired raw meter/RAPL integrals into the strict paired-v2 join schema."""
from __future__ import annotations

import argparse
from pathlib import Path

from greenshaping.energy.paired import pair_energy_records
from greenshaping.utils.io import read_csv, write_csv, write_json
from greenshaping.utils.provenance import file_sha256


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('choose a fresh measurement output file')
    rows = pair_energy_records(read_csv(args.raw))
    write_csv(rows, out)
    write_json({'raw_sha256': file_sha256(args.raw), 'paired_sha256': file_sha256(out), 'pairs': len(rows),
                'definition': '(shaped integral - matching idle integral) - (unshaped integral - matching idle integral)',
                'session_id': 'logical paired trial ID; acquisition IDs retained separately',
                'config_id': 'target shaping configuration for both members of the pair'}, out.with_suffix('.provenance.json'))


if __name__ == '__main__':
    main()
