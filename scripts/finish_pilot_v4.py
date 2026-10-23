"""Bounded post-processing supervisor; preserves failed runs and excludes secrets."""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from greenshaping.utils.io import read_json, write_json


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    a = p.parse_args()
    root = Path(a.root).resolve()
    write_json({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (root / 'scripts/finish_pilot_v4.py', root / 'scripts/summarize_pilot_v4.py')},
               root / 'postprocessing_sources.json')
    state = {'pid': os.getpid(), 'state': 'waiting_for_workflow'}
    deadline = time.monotonic() + 10800
    try:
        while time.monotonic() < deadline:
            workflow = read_json(root / 'workflow.json')
            if workflow['state'] == 'complete':
                break
            if workflow['state'] == 'failed':
                raise RuntimeError('workflow failed; all evidence retained, no final-test claims')
            write_json(state, root / 'postprocessing.json')
            time.sleep(10)
        else:
            raise TimeoutError('bounded three-hour wait expired')
        for name, command in [
            ('audit', [sys.executable, 'scripts/audit_pilot_v3.py', '--root', str(root / 'runs/adaptive_01'),
                       '--out', str(root / 'audit')]),
            ('summary', [sys.executable, 'scripts/summarize_pilot_v4.py', '--root', str(root), '--out', str(root / 'summary')]),
        ]:
            state.update(state='running', phase=name)
            write_json(state, root / 'postprocessing.json')
            with (root / f'{name}.log').open('wb') as log:
                subprocess.run(command, cwd=root, stdout=log, stderr=log, check=True)
        summary = read_json(root / 'summary/summary.json')
        audit = read_json(root / 'audit/report.json')
        quality = read_json(root / 'runs/adaptive_01/pilot_quality.json')
        assert all(v['sessions'] == 192 for v in quality['modes'].values())
        assert not quality['failures']
        capture_log = (root / 'runs/adaptive_01/tcpdump.log').read_text()
        drops = re.search(r'(\d+) packets dropped by kernel', capture_log)
        if drops is None or int(drops.group(1)) != 0:
            raise RuntimeError('capture drop count absent or nonzero; investigate preserved capture')
        assert not any(audit[k] for k in ('overflow_link_directions', 'cap_violations', 'io_errors'))
        assert audit['unique_wire_exact_fraction'] == 1.
        table = ['| Mode | Max advantage mean | Min | Max | TCP payload factor | Observed duration s |',
                 '|---|---:|---:|---:|---:|---:|']
        costs = {r['mode']: r for r in summary['observed_costs']}
        for row in summary['test_strongest_summary']:
            c = costs[row['mode']]
            table.append(f"| {row['mode']} | {row['mean']:.4f} | {row['min']:.4f} | {row['max']:.4f} | "
                         f"{c['observed_payload_factor']:.3f} | {c['observed_execution_median_s']:.4f} |")
        ci = summary['paired_question_bootstrap_first_seed']
        message = ('# Kết quả pilot thích ứng trên Mac Studio\n\n'
                   'Đã hoàn tất 960 phiên chính, 192 câu hỏi ở mỗi trong năm mode. '
                   'Đây là pilot synthetic trên một host, không phải kết quả chính thức của bài báo.\n\n'
                   + '\n'.join(table) + '\n\n'
                   'Maximum advantage lấy qua bốn attacker và ba secret. Mean/min/max ở bảng là qua năm seed '
                   'huấn luyện trên cùng capture; không phải lặp acquisition hay khoảng tin cậy. '
                   'Duration là cửa sổ traffic quan sát, gồm cover tail của DP, không phải analyst completion latency.\n\n'
                   f"Best single được chọn trên select: **{summary['best_single_from_select']}**. "
                   f"Khoảng bootstrap 95% theo question cho mức giảm maximum advantage của composition, seed {ci['seed']}: "
                   f"[{ci['reduction_ci95'][0]:.4f}, {ci['reduction_ci95'][1]:.4f}]. "
                   'Giá trị dương có lợi cho composition; khoảng chứa zero chưa cho thấy cải thiện rõ trên pilot này.\n\n'
                   'CNN được chọn từ hai ứng viên bằng train/select, khóa trước final test và gắn với hash pcap. '
                   'Toàn bộ query trace thích ứng đã được đối chiếu với local replay. '
                   f"Wire accounting khớp unique TCP payload ở {audit['wire_link_directions']} link/direction/session cases; "
                   f"{audit['duplicate_tcp_payload_bytes']} duplicate sequence bytes vẫn được giữ trong attacker/cost. "
                   'Đối chiếu byte không chứng nhận bảo vệ timing hoặc end-to-end DP.\n\n'
                   'Năng lượng chưa được đo. Các site có delay mô phỏng, dùng dữ liệu và bộ sinh synthetic chung; '
                   'không suy diễn tính khái quát trên tổ chức thật, hosted LLM, NetShaper hoặc Linux/edge. '
                   'Không điền các số này vào bảng kết quả chính của bản thảo.\n\n'
                   'Bằng chứng: summary/summary.json, selection/cnn_selection.json, evaluation/predictions.csv, '
                   'audit/report.json, configs/pilot_v4_protocol.json và source_hashes.json.\n')
        (root / 'pilot_v4_report.md').write_text(message)
        allowed = []
        for name in ('runs', 'selection', 'evaluation', 'audit', 'summary'):
            allowed.extend(p for p in (root / name).rglob('*') if p.is_file())
        allowed.extend(root / name for name in ('workflow.json', 'workflow.exit', 'tests.log', 'integration.log',
                                                'acquisition.log', 'selection.log', 'evaluation.log', 'audit.log', 'summary.log',
                                                'pilot_v4_report.md', 'source_hashes.json', 'postprocessing_sources.json',
                                                'configs/pilot_v4_protocol.json'))
        allowed = sorted(p for p in allowed if p.exists() and p.name not in ('key.pem', '.env') and p.suffix != '.key')
        hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in allowed}
        manifest = root / 'artifact_manifest.json'
        write_json(hashes, manifest)
        archive = root / 'pilot_v4_verified_results.tar.gz'
        state.update(state='running', phase='archive')
        write_json(state, root / 'postprocessing.json')
        with tarfile.open(archive, 'w:gz') as tf:
            for path in allowed + [manifest]:
                tf.add(path, arcname=str(Path('results') / path.relative_to(root)))
        state.update(state='complete', artifact_files=len(hashes), archive=str(archive),
                     archive_bytes=archive.stat().st_size, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        write_json(state, root / 'postprocessing.json')
        return 0
    except Exception as e:
        state.update(state='failed', error=str(e))
        write_json(state, root / 'postprocessing.json')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
