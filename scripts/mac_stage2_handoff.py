"""Finite local-only observer: write a morning handoff without changing acquisition."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def report(root, expired=False):
    results = root / 'results'
    state = read(root / 'queue/state.json')
    status = read(results / 'status.json')
    failure = read(results / 'failure.json')
    ready = read(results / 'readiness.json')
    backup = read(results / 'backup_audit.json')
    finished = state.get('queue_status') == 'idle_complete'
    checks = ready.get('engineering_checks_passed') is True
    complete = finished and status.get('stage') == 'completed' and checks and not failure
    terminal = finished or bool(failure) or expired
    evidence = {}
    for name in ['train', 'select', 'mistral_train', 'mistral_select', 'qwen14b_train', 'qwen14b_select']:
        evidence[name] = {'progress': read(results / name / 'progress.json'),
                          'summary': read(results / name / 'summary.json')}
    hashes = {}
    if terminal:
        # Small proof inventory; never copy private TLS keys or create raw captures.
        for f in results.rglob('*'):
            if f.is_file() and f.suffix in {'.json', '.jsonl', '.log'}:
                h = hashlib.sha256()
                with f.open('rb') as stream:
                    for b in iter(lambda: stream.read(1024 * 1024), b''):
                        h.update(b)
                hashes[str(f.relative_to(results))] = {'sha256': h.hexdigest(), 'bytes': f.stat().st_size}
    return {'updated_epoch': time.time(), 'run_root': str(root), 'terminal': terminal,
            'observer_deadline_expired': expired, 'pipeline_completed': complete,
            'stage': status.get('stage'), 'queue': state, 'failure': failure,
            'engineering_checks_passed': checks, 'backup_audit': backup,
            'engineering_selection': read(results / 'engineering_selection.json'),
            'evidence': evidence, 'result_proof_inventory': hashes,
            'cloud_used': False, 'test_used': False, 'start_EC2_automatically': False,
            'ready_for_large_cloud_acquisition': False, 'main_paper_ready': False,
            'next_action': ('Review evidence and arrange a bounded real-network preflight before a large run.'
                            if complete else 'Keep EC2 stopped; review failed or incomplete stages before acquisition.'),
            'pending': ['Current EC2 IPs and exclusive resource window, especially T4',
                        'Real-network clocks, capture lifecycle and packet-drop preflight',
                        'Matched observer data for frozen attacker/privacy evaluation',
                        'Physical energy evidence if joule claims are retained',
                        '10 Gbps is a target, not an achieved result']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--deadline-epoch', type=float, required=True)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    out = a.root / 'handoff'; out.mkdir(exist_ok=True)
    while True:
        expired = time.time() >= a.deadline_epoch
        d = report(a.root, expired)
        write(out / 'morning_status.json', d)
        outcome = 'ĐÃ HOÀN TẤT kiểm tra kỹ thuật Mac' if d['pipeline_completed'] else 'CHƯA ĐỦ ĐIỀU KIỆN chuyển sang chạy lớn'
        text = '\n'.join(['# Green Shaping — bàn giao buổi sáng', '', outcome, '',
            'Giai đoạn gần nhất: ' + str(d['stage']),
            'Kết quả chi tiết: morning_status.json; dữ liệu gốc: ../results/.',
            'Bằng chứng lượt R08 thất bại được giữ riêng, không trộn với R09.', '',
            '## Việc tiếp theo', '', *('- ' + s for s in d['pending']), '',
            'Không tự bật EC2. Chỉ chạy một lượt preflight có giới hạn sau khi kiểm tra bằng chứng.',
            'Kết quả Mac chỉ kiểm tra chức năng/chi phí plaintext và tính nhất quán câu trả lời; không chứng minh privacy, joule hoặc 10 Gbps.',
            'Nếu có lỗi: giữ log, ledger và snapshot; không tự lặp đo cho đến khi đạt.', ''])
        (out / 'MORNING_HANDOFF.md').write_text(text)
        if d['terminal'] or a.once:
            return
        time.sleep(min(60, max(1, a.deadline_epoch - time.time())))


if __name__ == '__main__':
    main()
