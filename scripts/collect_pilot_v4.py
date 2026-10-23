"""Mirror the isolated pilot status, then download and verify its completed archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    remote = '/Users/sonha/Projects/GreenShaping/run_v4_20261020_r2'
    helper = [sys.executable, str(code / 'scripts/remote_runner.py'), '--known-hosts',
              str(code / 'outputs/remote_20261020/known_hosts')]
    payload = f'''import json
from pathlib import Path
root=Path({remote!r})
names=['workflow.json','postprocessing.json','runs/integration_01/pilot_status.json',
       'runs/adaptive_01/pilot_status.json','evaluation/status.json']
result={{}}
for name in names:
    p=root/name
    if p.exists():
        try: result[name]=json.loads(p.read_text())
        except json.JSONDecodeError: pass
print(json.dumps(result))
'''
    deadline, failures = time.monotonic() + 10800, 0
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(helper + ['exec', 'python3 -'], input=payload, text=True,
                                    capture_output=True, check=True, timeout=90)
            progress = json.loads(result.stdout)
            (out / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
            final = progress.get('postprocessing.json', {})
            if any(progress.get(name, {}).get('state') == 'failed' for name in ('workflow.json', 'postprocessing.json')):
                raise RuntimeError('remote workflow/postprocessing failed; inspect progress.json and preserved remote logs')
            if final.get('state') == 'complete':
                break
            failures = 0
        except (subprocess.SubprocessError, json.JSONDecodeError) as e:
            failures += 1
            print(f'transient monitor failure {failures}: {type(e).__name__}', flush=True)
            if failures >= 10:
                raise
        time.sleep(30)
    else:
        raise TimeoutError('bounded three-hour monitor expired')
    archive = out / 'pilot_v4_verified_results.tar.gz'
    subprocess.run(helper + ['get', final['archive'], str(archive)], check=True, timeout=900)
    assert archive.stat().st_size == final['archive_bytes']
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    if sha != final['archive_sha256']:
        raise ValueError('download checksum mismatch')
    with tarfile.open(archive) as tf:
        if any(Path(m.name).name in ('key.pem', '.env') or Path(m.name).suffix == '.key' for m in tf.getmembers()):
            raise ValueError('unexpected secret filename in archive')
        tf.extractall(out, filter='data')
    hashes = json.loads((out / 'results/artifact_manifest.json').read_text())
    for name, expected in hashes.items():
        if hashlib.sha256((out / 'results' / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'artifact checksum mismatch: {name}')
    verification = {'state': 'complete', 'archive_sha256': sha, 'archive_bytes': archive.stat().st_size,
                    'verified_files': len(hashes), 'remote_root': remote, 'energy_measured': False}
    (out / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    shutil.copyfile(out / 'results/pilot_v4_report.md', code / 'docs/pilot_v4_results_20261020.md')
    print(json.dumps(verification), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
