"""Locate historical cross-observer residuals relative to session boundaries."""
from pathlib import Path
import json
from greenshaping.data.capture_windows import account
from greenshaping.data.wire_capture import packets

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'outputs/results_review_20261021/matched_raw/results'
x=json.loads((ROOT/'outputs/offline_resilience_20261021/matched_05_analysis_inputs.json').read_text())
allout={}
for mode in ['dp','rounds']:
    rows=x['sessions'][mode+'_session_evidence.tar.gz']
    windows=[dict(id=str(i),start_epoch=r['start_epoch'],end_epoch=r['end_epoch']) for i,r in enumerate(rows)]
    sides={n:account(packets(R/f'{n}_{mode}_traffic.pcap',('172.31.34.237','172.31.38.41'),fingerprints=True),windows) for n in ['T2','T3']}
    a,b=sides['T2']['fingerprints'],sides['T3']['fingerprints']
    keys=set(a-b)|set(b-a)
    observations={n:[] for n in sides}
    bounds=[v for w in windows for v in [w['start_epoch'],w['end_epoch']]]
    for n in sides:
        for row in packets(R/f'{n}_{mode}_traffic.pcap',('172.31.34.237','172.31.38.41'),fingerprints=True):
            if row['fingerprint'] in keys:
                t=row['epoch']
                row['nearest_boundary_delta_us']=min([(t-v)*1e6 for v in bounds],key=abs)
                row['in_session']=any(w['start_epoch']<=t<w['end_epoch'] for w in windows)
                observations[n].append(row)
    allout[mode]=observations
    print(mode,{n:[(r['network_bytes'],round(r['nearest_boundary_delta_us'],1),r['in_session']) for r in rs] for n,rs in observations.items()},flush=True)
(ROOT/'outputs/results_review_20261021/packet_reconciliation/residual_diagnostics.json').write_text(json.dumps(allout,indent=2))
