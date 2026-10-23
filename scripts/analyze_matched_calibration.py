"""Descriptive reanalysis of fixed TRAIN calibration; never acquires or tunes."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'outputs/offline_resilience_20261021'
OUT = ROOT / 'outputs/results_review_20261021'
MODES = ['none', 'pad', 'rounds', 'dp', 'combo', 'netshaper_no_shaping', 'netshaper_shaping']


def main():
    OUT.mkdir(exist_ok=True)
    inputs = INPUT / 'matched_05_analysis_inputs.json'
    aligned = INPUT / 'matched_05_aligned_capture_analysis.json'
    x, a = json.loads(inputs.read_text()), json.loads(aligned.read_text())
    baseline = a['none']['T2']['session_window_bytes']
    rows, details = [], {}
    rng = np.random.default_rng(20261021)
    original = x['sessions']['none_session_evidence.tar.gz']
    base = {(r['question_id'], r['repetition']): r for r in original}
    for mode in MODES:
        sessions = x['sessions'][mode + '_session_evidence.tar.gz']
        assert len(sessions) == 24 and all(r['passed'] for r in sessions)
        phase = x['phases'][mode + '_summary.json']
        assert phase['replica_acknowledged_sessions'] == 24
        qids = sorted({r['question_id'] for r in sessions})
        assert len(qids) == 12 and {(r['question_id'], r['repetition']) for r in sessions} == set(base)
        tool = np.array([r['tool_completed_epoch'] - r['start_epoch'] for r in sessions])
        end = np.array([r['end_epoch'] - r['start_epoch'] for r in sessions])
        execution = np.array([r['tool_completed_epoch'] - r['transport']['epoch'] for r in sessions])
        grouped = np.array([np.mean([
            (r['tool_completed_epoch'] - r['start_epoch']) -
            (base[(q, r['repetition'])]['tool_completed_epoch'] - base[(q, r['repetition'])]['start_epoch'])
            for r in sessions if r['question_id'] == q]) for q in qids])
        ci = np.percentile(grouped[rng.integers(0, len(qids), size=(10000, len(qids)))].mean(axis=1), [2.5, 97.5])
        t2, t3 = a[mode]['T2'], a[mode]['T3']
        row = dict(mode=mode, sessions=24, distinct_train_questions=12,
                   tool_p50_s=float(np.median(tool)), tool_p95_s=float(np.percentile(tool, 95)),
                   tool_after_transport_epoch_p50_s=float(np.median(execution)),
                   session_end_p50_s=float(np.median(end)),
                   ip_MB_in_session_windows=t2['session_window_bytes'] / 1e6,
                   ip_MB_mean_per_session=t2['session_window_bytes'] / 24e6,
                   ip_byte_factor=t2['session_window_bytes'] / baseline,
                   t2_t3_session_byte_difference=t3['session_window_bytes'] - t2['session_window_bytes'],
                   paired_mean_tool_delay_s=float(grouped.mean()),
                   paired_question_bootstrap_low_s=float(ci[0]), paired_question_bootstrap_high_s=float(ci[1]))
        rows.append(row)
        details[mode] = dict(raw_capture=x['captures']['phase_ledger/committed/' + mode + '.json']['record']['captures'],
                             aligned_capture=a[mode], tool_latency_samples_s=tool.tolist(), session_end_samples_s=end.tolist())
    result = dict(scope='Descriptive TRAIN calibration only; 12 fixed questions x 2 repetitions per mode',
                  sessions=168, modes=rows, details=details,
                  cautions=['No attacker evaluation on these public-data cloud captures',
                            'No measured physical joules, no matched privacy or energy baseline',
                            'One physical data endpoint, 20 logical partitions, one multiplexed observer link',
                            'One first tool round per question, no LLM planning or final answer evaluation',
                            'Latency from start includes a configured 1-second scheduling lead',
                            'Byte counts include IPv4 TCP/UDP headers and ACKs, exclude Ethernet and other deployment links',
                            'Session-window byte cost excludes between-session and deployment setup/teardown cover; not steady-state cost',
                            'Baseline already has all-site control frames: 4 x 20 x 4113 = 329040 plaintext framed bytes/session, about 95.33% of observed baseline IPv4 bytes; do not treat it as minimal application-only baseline',
                            'Modes executed in separate randomized-order phases, not interleaved per-question paired trials',
                            'Bootstrap is exploratory over 12 questions, not confirmatory or a phase/time confounding correction'],
                  input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (inputs, aligned)})
    (OUT / 'metrics.json').write_text(json.dumps(result, indent=2))
    with (OUT / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    labels = ['None', 'Padding', 'Rounds', 'DP arrivals', 'Combo', 'NetShaper off', 'NetShaper on']
    axes[0].barh(labels, [r['ip_byte_factor'] for r in rows], color='#386da8')
    axes[0].set_xscale('log');axes[0].set_xlabel('IPv4 byte factor vs none (log scale)')
    axes[0].set_title('Aligned session windows, T2 observer')
    for i, r in enumerate(rows):axes[0].text(r['ip_byte_factor']*1.08,i,f"{r['ip_byte_factor']:.2f}x",va='center',fontsize=9)
    axes[0].set_xlim(.8, 2500)
    axes[1].barh(labels, [r['tool_p50_s'] for r in rows], color='#278371', label='Tool result p50')
    axes[1].plot([r['session_end_p50_s'] for r in rows], range(7), 'o',color='#d07828',label='Session end p50')
    axes[1].set_xlabel('Seconds from client start (includes 1 s lead)');axes[1].legend(loc='lower right')
    axes[1].set_title('Correct first tool round; no LLM answer latency')
    fig.suptitle('EC2 TRAIN calibration: 12 questions x 2 repeats per mode; no privacy/energy claim')
    fig.savefig(OUT / 'calibration_costs.png', dpi=180);plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
