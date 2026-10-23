"""Read-only packet reconciliation of retained calibration; no coverage inference."""
import argparse
import gc
import json
from pathlib import Path
from collections import Counter
from greenshaping.data.capture_windows import account, compare
from greenshaping.data.wire_capture import packets
from greenshaping.utils.evidence import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    x=json.loads(a.inputs.read_text());summary={}
    for name,rows in x['sessions'].items():
        mode=name.removesuffix('_session_evidence.tar.gz')
        windows=[dict(id=f"{r['question_id']}_{r['repetition']}",start_epoch=r['start_epoch'],end_epoch=r['end_epoch']) for r in rows]
        sides={n:account(packets(a.results/f'{n}_{mode}_traffic.pcap',('172.31.34.237','172.31.38.41'),fingerprints=True),windows) for n in ['T2','T3']}
        diagnostic=compare(sides['T2'],sides['T3']);diagnostic['missing_lifecycle_proof']=True
        diagnostic['note']='Historical acquisition recorded packet validity, not coordinated capture-ready/stopped timestamps. Cannot retroactively certify strict coverage.'
        for n in sides:sides[n].pop('fingerprints')
        summary[mode]={'comparison':diagnostic,'windows':sides}
        atomic_json(a.out/(mode+'.json'),summary[mode]);print(mode,diagnostic,flush=True);gc.collect()
    atomic_json(a.out/'summary.json',summary)


if __name__=='__main__':main()
