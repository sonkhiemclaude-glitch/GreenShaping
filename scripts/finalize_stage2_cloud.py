"""Finite local postprocessing of a terminal EC2 validation; never starts cloud work."""
import argparse
import hashlib
import json
import re
import statistics
import tarfile
import time
from pathlib import Path
from greenshaping.data.capture_windows import account, compare
from greenshaping.data.wire_capture import packets
from greenshaping.utils.evidence import atomic_json


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--deadline-epoch',type=float,required=True);a=p.parse_args()
    r=a.root;out=r/'review';out.mkdir(exist_ok=False)
    while read(r/'queue/state.json').get('queue_status')!='idle_complete':
        if time.time()>a.deadline_epoch:
            atomic_json(out/'status.json',{'stage':'queue_not_terminal_at_deadline','main_paper_ready':False});return 2
        time.sleep(15)
    result=r/'results';summary={};manifest=read(result/'manifest.json')
    if not manifest:raise ValueError('missing terminal acquisition manifest')
    if any(digest(result/n)!=h for n,h in manifest.items()):raise ValueError('acquisition manifest hash mismatch')
    for file in sorted(result.glob('*_session_evidence.tar.gz')):
        mode=file.name.removesuffix('_session_evidence.tar.gz');rows=[]
        with tarfile.open(file) as t:
            for member in t:
                if re.fullmatch(r'client/session_\d+\.json',member.name):rows.append(json.load(t.extractfile(member)))
        rows.sort(key=lambda d:d['start_epoch'])
        if not rows:continue
        windows=[{'id':str(v['question_id'])+'_'+str(v['repetition']),'start_epoch':v['start_epoch'],'end_epoch':v['end_epoch']} for v in rows]
        lifecycle=read(result/(mode+'_capture_lifecycle.json'));sides={};fingerprints={}
        for node in ['T2','T3']:
            log=(result/(node+'_'+mode+'_tcpdump.log')).read_text()
            match=re.search(r'(\d+) packets dropped by kernel',log)
            info=dict(lifecycle.get(node,{}));info['kernel_dropped_packets']=int(match[1]) if match else None
            pcap=result/(node+'_'+mode+'_traffic.pcap')
            counted=account(packets(pcap,('172.31.34.237','172.31.38.41'),fingerprints=True),windows,lifecycle=info)
            fingerprints[node]=counted
            sides[node]={k:v for k,v in counted.items() if k!='fingerprints'}|{'lifecycle':info,'pcap_sha256':digest(pcap),'pcap_bytes':pcap.stat().st_size}
        agreement=compare(fingerprints['T2'],fingerprints['T3'])
        times=[v['tool_completed_epoch']-v['start_epoch'] for v in rows if 'tool_completed_epoch' in v]
        summary[mode]={'sessions':len(rows),'passed':sum(bool(v['passed']) for v in rows),'captures':sides,'comparison':agreement,'tool_p50_s':statistics.median(times) if times else None,'mean_T2_IP_bytes':sides['T2']['session_bytes']/len(rows)}
        atomic_json(out/(mode+'.json'),summary[mode])
    complete=read(result/'summary.json').get('all_passed') is True
    cleanup_ok=read(result/'cleanup.json').get('errors')==[]
    coverage=bool(summary) and all(v['lifecycle_coverage_verified'] for s in summary.values() for v in s['captures'].values())
    report={'acquisition_complete':complete,'cleanup_verified':cleanup_ok,'manifest_verified':True,'all_capture_lifecycle_coverage_verified':coverage,'sessions':sum(v['sessions'] for v in summary.values()),'passed':sum(v['passed'] for v in summary.values()),'modes':summary,'failure':read(result/'failure.json'),'main_paper_ready':False,'large_acquisition_ready':False,'scope':'TRAIN descriptive three-step workflow; multiplexed observer, no physical energy or attacker result','instance_stop_required':True}
    atomic_json(out/'summary.json',report)
    lines=['Green Shaping: bounded EC2 stage2 batch finished.','Sessions: '+str(report['passed'])+'/'+str(report['sessions']),'Acquisition complete: '+str(complete),'Cleanup verified: '+str(cleanup_ok),'Lifecycle coverage: '+str(coverage),'Review summary.json; preserve all failed trials.','No physical joule/privacy/final-test claim. Stop EC2 instances when no authorized follow-up uses them.']
    (out/'BATCH_NOTICE.txt').write_text('\n'.join(lines)+'\n')
    # One compact controller handoff; raw pcaps stay in their original paths.
    archive=r/'stage2_review_metadata.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for base in [out,result]:
            for f in base.rglob('*'):
                if f.is_file() and f.suffix in {'.json','.jsonl','.log','.txt','.gz'}:
                    t.add(f,arcname=str(f.relative_to(r)),recursive=False)
    atomic_json(r/'stage2_review_metadata.sha256.json',{'sha256':digest(archive),'bytes':archive.stat().st_size})
    atomic_json(out/'status.json',{'stage':'completed','main_paper_ready':False})
    return 0


if __name__=='__main__':raise SystemExit(main())
