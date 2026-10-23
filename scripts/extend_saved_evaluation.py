"""Read-only retrospective analysis; never accesses final-test data."""
import bisect,csv,hashlib,json,tarfile
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score,accuracy_score,confusion_matrix
from greenshaping.data.wire_capture import packets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
MODES=['none','pad','rounds','dp','combo','netshaper_no_shaping','netshaper_shaping'];SEED=20261023
OUT=Path('outputs/evaluation_extension_20261022');BASE=Path('outputs/ec2_final_preflight_20261022')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def csvout(name,rows):
 with (OUT/(name+'.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run():
 OUT.mkdir(exist_ok=True)
 plan={'seed':SEED,'scope':'retrospective exploratory grouped cross-validation; 12 questions, not confirmatory','features':'IPv4 bytes, packets, directional bytes; packet-size mean/std/p10/p50/p90; observed span; IAT mean/std/p90; 22 half-second byte bins','family_folds':'leave one requested size out, 3 folds','size_folds':'leave one family out, 4 folds','RF':{'n_estimators':100,'max_depth':4,'min_samples_leaf':2,'class_weight':'balanced','n_jobs':2},'selection':'no hyperparameter selection; fixed before this diagnostic','excluded':['payload','absolute epoch','question ID features','driver tool-completion feature','final-test bank','per-site participation attacks'],'window_assumption':'oracle-delimited retained windows; segmentation not learned','uncertainty':'4000 question-cluster resamples; conditional on acquired questions'}
 (OUT/'diagnostic_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 bank={q['question_id']:q for q in map(json.loads,(BASE/'train_n20.jsonl').open())};review=json.loads((BASE/'verified/review/summary.json').read_text());inputs=[BASE/'train_n20.jsonl',BASE/'verified/review/summary.json'];rows=[];features=[];scores=[];predictions=[];folds=[]
 for mode in MODES:
  arc=BASE/'verified/results'/(mode+'_session_evidence.tar.gz');inputs.append(arc)
  with tarfile.open(arc) as t:ss=[json.load(t.extractfile(m)) for m in t if m.name.startswith('client/session_') and m.name.endswith('.json')]
  ss.sort(key=lambda s:s['start_epoch']);assert len(ss)==24 and all(s['passed'] for s in ss)
  cap=review['modes'][mode]['captures']['T2'];net={r['id']:r for r in cap['per_session']};obs=[[] for s in ss];starts=[s['start_epoch'] for s in ss];pcap=BASE/'raw'/('T2_'+mode+'_traffic.pcap');inputs.append(pcap);assert sha(pcap)==cap['pcap_sha256']
  for p in packets(pcap,('172.31.34.237','172.31.38.41')):
   i=bisect.bisect_right(starts,p['epoch'])-1
   if i>=0 and p['epoch']<=ss[i]['end_epoch']:obs[i].append((p['epoch']-starts[i],p['network_bytes']))
  for s,oo in zip(ss,obs):
   q=bank[s['question_id']];assert q['metadata']['partition']=='train';n=net[f"{s['question_id']}_{s['repetition']}"];assert sum(v for _,v in oo)==n['network_bytes']
   f=s['transport']['framed_plaintext_bytes'];control=f.get('sent_control',0)+f.get('received_control',0);data=f.get('sent_data',0)+f.get('received_data',0)
   r=dict(mode=mode,question_id=s['question_id'],repetition=s['repetition'],family=q['family'],requested_size=q['metadata']['requested_size'],**{k:n[k] for k in ['network_bytes','packets','up_bytes','down_bytes']},tool_s=s['tool_completed_epoch']-s['start_epoch'],session_s=s['end_epoch']-s['start_epoch'],framed_control_bytes=control,framed_data_bytes=data,framed_control_fraction=control/max(control+data,1));rows.append(r)
   tt=np.array([t for t,b in oo]);bb=np.array([b for t,b in oo]);dt=np.diff(tt);bins=np.histogram(tt,bins=np.arange(23)*.5,weights=bb)[0]
   x=[r[k] for k in ['network_bytes','packets','up_bytes','down_bytes']]+[bb.mean(),bb.std(),*np.quantile(bb,[.1,.5,.9]),tt[-1]-tt[0],dt.mean(),dt.std(),np.quantile(dt,.9)]+list(bins)
   features.append(dict(mode=mode,question_id=r['question_id'],repetition=r['repetition'],family=r['family'],requested_size=r['requested_size'],features=list(map(float,x))))
  rr=[r for r in features if r['mode']==mode];X=np.log1p([r['features'] for r in rr]);assert np.isfinite(X).all()
  for target,group in [('family','requested_size'),('requested_size','family')]:
   y=np.array([r[target] for r in rr]);g=np.array([r[group] for r in rr]);classes=sorted(set(y))
   for ablation in ['bytes_only','all_metadata']:
    xx=X[:,:1] if ablation=='bytes_only' else X
    for classifier in ['majority','rf']:
     pred=np.empty_like(y)
     for label in sorted(set(g)):
      te=g==label;tr=~te;tq=sorted({r['question_id'] for r,t in zip(rr,tr) if t});vq=sorted({r['question_id'] for r,t in zip(rr,te) if t});assert not set(tq)&set(vq);assert set(y[tr])==set(classes)
      model=DummyClassifier(strategy='most_frequent') if classifier=='majority' else RandomForestClassifier(n_estimators=100,max_depth=4,min_samples_leaf=2,class_weight='balanced',random_state=SEED,n_jobs=2)
      model.fit(xx[tr],y[tr]);pred[te]=model.predict(xx[te]);folds.append(dict(mode=mode,target=target,ablation=ablation,classifier=classifier,held_out=label,train_questions=tq,validation_questions=vq))
     scores.append(dict(mode=mode,target=target,ablation=ablation,classifier=classifier,questions=12,sessions=24,balanced_accuracy=balanced_accuracy_score(y,pred),accuracy=accuracy_score(y,pred),chance_balanced_accuracy=1/len(classes),confusion_matrix=json.dumps(confusion_matrix(y,pred,labels=classes).tolist()),classes=json.dumps(classes)))
     predictions.extend(dict(mode=mode,target=target,ablation=ablation,classifier=classifier,question_id=r['question_id'],repetition=r['repetition'],truth=str(a),prediction=str(b)) for r,a,b in zip(rr,y,pred))
  print('analysed',mode,flush=True)
 qs=sorted({r['question_id'] for r in rows});baseline=np.array([np.mean([r['network_bytes'] for r in rows if r['mode']=='none' and r['question_id']==q]) for q in qs]);summary=[];strata=[];rng=np.random.default_rng(SEED);idx=rng.integers(0,len(qs),(4000,len(qs)))
 for mode in MODES:
  rr=[r for r in rows if r['mode']==mode];v=np.array([np.mean([r['network_bytes'] for r in rr if r['question_id']==q]) for q in qs]);lo,hi=np.quantile(v[idx].mean(axis=1)/baseline[idx].mean(axis=1),[.025,.975]);cap=review['modes'][mode]['captures']['T2'];outside=cap['outside_session_bytes'];inside=cap['session_bytes']
  summary.append(dict(mode=mode,questions=12,sessions=24,mean_IP_MB=v.mean()/1e6,ratio_to_controls=v.mean()/baseline.mean(),ratio_cluster_ci_lower=lo,ratio_cluster_ci_upper=hi,tool_p50_s=np.median([r['tool_s'] for r in rr]),session_p50_s=np.median([r['session_s'] for r in rr]),mean_framed_control_fraction=np.mean([r['framed_control_fraction'] for r in rr]),outside_MB=outside/1e6,outside_fraction_total=outside/(inside+outside),capture_total_MB_per_session=(inside+outside)/24/1e6))
  for group in ['family','requested_size']:
   for label in sorted({r[group] for r in rr}):
    sub=[r for r in rr if r[group]==label];strata.append(dict(mode=mode,group=group,label=label,questions=len({r['question_id'] for r in sub}),sessions=len(sub),mean_IP_MB=np.mean([r['network_bytes'] for r in sub])/1e6,median_tool_s=np.median([r['tool_s'] for r in sub]),mean_control_fraction=np.mean([r['framed_control_fraction'] for r in sub])))
 for name,data in [('session_costs',rows),('stratified_costs',strata),('cost_summary',summary),('attacker_scores',scores),('attacker_predictions',predictions)]:csvout(name,data)
 for name,data in [('attacker_folds',folds),('features',features),('input_hashes',{str(p):sha(p) for p in inputs})]:(OUT/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
 fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained');short=['None','Pad','Rounds','DP','Combo','Net off','Net on']
 for ax,target in zip(axes,['family','requested_size']):
  for ablation,marker in [('bytes_only','o'),('all_metadata','s')]:ax.plot(range(7),[next(r['balanced_accuracy'] for r in scores if r['mode']==m and r['target']==target and r['ablation']==ablation and r['classifier']=='rf') for m in MODES],marker=marker,label=ablation.replace('_',' '))
  ax.axhline(.25 if target=='family' else 1/3,color='gray',linestyle='--',label='chance BA');ax.set(ylim=(0,1.05),title=target.replace('_',' '),ylabel='Exploratory balanced accuracy');ax.set_xticks(range(7),short,rotation=35);ax.legend(fontsize=7)
 fig.savefig(OUT/'attacker_diagnostic.pdf');fig.savefig(OUT/'attacker_diagnostic.png',dpi=140);plt.close(fig)
 (OUT/'network_analysis_complete.json').write_text(json.dumps({'complete':True,'sessions':len(rows),'unique_questions':12,'final_test_opened':False,'separate_site_attacks_supported':False,'scope':'retrospective grouped cross-validation; not confirmatory'},indent=2)+'\n')
if __name__=='__main__':run()
