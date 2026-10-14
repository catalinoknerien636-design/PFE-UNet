from pathlib import Path
import csv,hashlib,json,math,statistics
R=Path('<LOCAL_HOST_PATH_06>');H=R/'83_capacity_compute_evidence';A=H/'analysis_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
c=read(H/'contract.json')
for p,h in c['sources'].items():assert digest(Path(p))==h
results=read(A/'capacity_compute_timing.json');independent=[]
for model in results:
 detail=read(A/(model['method']+'_detail.json'));mac=0
 for op in detail['convolutions']:
  n,co,ho,wo=op['output_shape'];mac+=n*ho*wo*co*op['in_channels']*op['kernel'][0]*op['kernel'][1]//op['groups']
 assert mac==model['conv_MACs_per_image']
 selected=[x for x in c['runs'] if x['config']['method']==model['method']];tr=[];va=[]
 for x in selected:
  history=rows(Path(x['config']['output_dir'])/'history.csv');assert len(history)==50
  tr.append(math.fsum(float(r['train_seconds']) for r in history)/3600);va.append(math.fsum(float(r['validation_seconds']) for r in history)/3600)
 for key,vals in [('train_hours',tr),('validation_hours',va),('loop_hours',[x+y for x,y in zip(tr,va)])]:
  for stat,value in [('mean',statistics.mean(vals)),('sample_sd',statistics.stdev(vals)),('minimum',min(vals)),('maximum',max(vals))]:assert math.isclose(value,model[key+'_'+stat],rel_tol=0,abs_tol=1e-12)
 independent.append({'method':model['method'],'convolution_MACs':mac,'five_full_histories_recomputed':True})
p=next(x for x in results if x['method']=='pfe_reference');w=next(x for x in results if x['method']=='wide_direct');d=next(x for x in results if x['method']=='direct_reference')
summary={'passed':True,'models':independent,'sources':len(c['sources']),'relative_parameter_excess_wide_to_pfe':w['trainable_parameters']/p['trainable_parameters']-1,'pfe_conv_MAC_reduction_vs_wide':1-p['conv_MACs_per_image']/w['conv_MACs_per_image'],'pfe_conv_MAC_increase_vs_direct':p['conv_MACs_per_image']/d['conv_MACs_per_image']-1,'all_1250_epochs':True,'GPU_or_prediction_replay':False}
(H/'independent_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(H/'writing_outline.md').write_text('''# Capacity and computational cost outline

- Question: the completed width control matches parameter count approximately; does that also match arithmetic cost?
- Evidence: frozen five PCOD U-Net architectures, 1x4x352x352 meta graph; all25 histories/1250epochs; existing corrected test summaries.
- Definitions: connected meta-gradient parameters are graph reachability, not nonzero empirical gradients or learned degrees of freedom. Dense Conv2d MACs exclude pointwise, pooling, normalization, interpolation and hardware execution details. One GMAC is10^9 pairs.
- Result: PFE12.739313M/34.720874496GMAC; widened Direct12.883841M/43.302465536GMAC; ordinary Direct7.849889M/26.410385408GMAC.
- Interpretation: near parameter matching still differs in convolution arithmetic. PFE has more cost than ordinary Direct and less than widened Direct; neither establishes superior accuracy or practical efficiency.
- Historical loop durations: retain all seeds and epochs. Describe train and validation separately, no throughput inference; no independent latency, energy, memory or statistical ranking.
- Manuscript insertion: main results paragraph plus SI S19/TableS25; no change to reported segmentation results, statistical claims, historical tables, or bibliography.
''',encoding='utf-8')
print(json.dumps(summary))
