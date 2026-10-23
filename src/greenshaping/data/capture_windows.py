"""Half-open session accounting, with explicit capture-lifecycle coverage.

Traffic timestamps alone cannot prove that a quiet capture was running. Lifecycle
proofs come from the acquisition controller; missing proofs are never inferred.
"""
import bisect
import math
from collections import Counter


def validate_windows(windows):
    rows = sorted(windows, key=lambda w: w['start_epoch'])
    seen = set()
    previous = -math.inf
    for w in rows:
        identity = str(w['id'])
        start, end = w['start_epoch'], w['end_epoch']
        if identity in seen or not all(math.isfinite(t) for t in (start, end)) or start >= end or start < previous:
            raise ValueError('unique IDs and finite nonoverlapping start < end windows required')
        previous = end
        seen.add(identity)
    if not rows:
        raise ValueError('session windows required')
    return rows


def account(rows, windows, lifecycle=None):
    windows = validate_windows(windows)
    starts = [w['start_epoch'] for w in windows]
    per = [dict(id=w['id'], network_bytes=0, packets=0, up_bytes=0, down_bytes=0) for w in windows]
    outside, phase, total = 0, 0, 0
    signatures = Counter()
    for row in rows:
        t, size = row['epoch'], row['network_bytes']
        if not math.isfinite(t) or size < 0 or row['direction'] not in ('up', 'down'):
            raise ValueError('invalid packet observation')
        total += size
        i = bisect.bisect_right(starts, t) - 1
        if starts[0] <= t < windows[-1]['end_epoch']:
            phase += size
        if i >= 0 and t < windows[i]['end_epoch']:
            per[i]['network_bytes'] += size
            per[i]['packets'] += 1
            per[i][row['direction'] + '_bytes'] += size
            if row.get('fingerprint'):
                signatures[row['fingerprint']] += 1
        else:
            outside += size
    coverage = bool(lifecycle and lifecycle.get('capture_ready_epoch', math.inf) <= starts[0]
                    and lifecycle.get('capture_stopped_epoch', -math.inf) >= windows[-1]['end_epoch']
                    and lifecycle.get('kernel_dropped_packets') == 0)
    return dict(per_session=per, session_bytes=sum(w['network_bytes'] for w in per),
                phase_bytes=phase, outside_session_bytes=outside, total_bytes=total,
                lifecycle_coverage_verified=coverage, fingerprints=signatures)


def compare(left, right):
    a, b = left['fingerprints'], right['fingerprints']
    if not a or not b:
        raise ValueError('packet fingerprints required for reconciliation')
    return dict(left_only_packets=sum((a-b).values()), right_only_packets=sum((b-a).values()),
                matching_packets=sum((a & b).values()),
                byte_difference=right['session_bytes'] - left['session_bytes'],
                exact_packet_match=a == b,
                strict_comparison_passed=a == b and left['lifecycle_coverage_verified']
                                        and right['lifecycle_coverage_verified'])
