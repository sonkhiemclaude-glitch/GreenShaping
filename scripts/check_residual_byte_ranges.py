"""Diagnose TCP payload coverage without calling packetization differences loss."""
import json
from pathlib import Path
from greenshaping.data.wire_capture import packets
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'outputs/results_review_20261021/packet_reconciliation'
d=json.loads((P/'residual_diagnostics.json').read_text())['dp']
targets=[dict(r,overlaps=[]) for r in d['T3'] if r.get('tcp_payload_bytes',0)>0]
for row in packets(ROOT/'outputs/results_review_20261021/matched_raw/results/T2_dp_traffic.pcap',('172.31.34.237','172.31.38.41'),fingerprints=True):
    if row.get('tcp_payload_bytes',0)<=0:continue
    for t in targets:
        if (row['src'],row['dst'],row['source_port'],row['destination_port'])!=(t['src'],t['dst'],t['source_port'],t['destination_port']):continue
        delta=(row['tcp_seq']-t['tcp_seq'])%(2**32)
        if delta>2**31:delta-=2**32
        lo=max(0,delta);hi=min(t['tcp_payload_bytes'],delta+row['tcp_payload_bytes'])
        if hi>lo:t['overlaps'].append({'start':lo,'end':hi,'epoch':row['epoch'],'ip_id':row['ip_id'],'fingerprint':row['fingerprint']})
for t in targets:
    end=covered=0
    for row in sorted(t['overlaps'],key=lambda r:r['start']):
        if row['end']>end:covered+=row['end']-max(end,row['start']);end=row['end']
    t['covered_payload_bytes']=covered;t['payload_covered_at_T2']=covered==t['tcp_payload_bytes']
(P/'dp_byte_range_reconciliation.json').write_text(json.dumps(targets,indent=2))
print(json.dumps({'data_packets':len(targets),'fully_covered_at_other_observer':sum(t['payload_covered_at_T2'] for t in targets),
                  'uncovered_payload_bytes':sum(t['tcp_payload_bytes']-t['covered_payload_bytes'] for t in targets)}))
