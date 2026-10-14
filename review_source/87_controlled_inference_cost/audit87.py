from pathlib import Path
import sys,statistics,math,os,struct,hashlib,collections
H=Path(__file__).resolve().parent;R=H.parent
sys.path.insert(0,str(R/'80_pgsnet_module_qualification'));import qualify80 as q
sys.path.insert(0,str(R/'84_worker_supervisor_qualification'));import win84_v4 as w

def quantile(xs,p):
 ys=sorted(xs);position=(len(ys)-1)*p;lo=math.floor(position);hi=math.ceil(position)
 return ys[lo]+(ys[hi]-ys[lo])*(position-lo)

def audit(phase):
 c=q.read(H/'contract.json');q.verify(c);seal=q.a.digest(H/'contract.json')
 assert seal=='ec54c73c7370eb274605e29eef31547b9fec4a88c62dc4263bbc95b34fc59064'
 out=H/(phase+'_v1');complete=q.read(out/'complete.json');order=c['qualification_order'] if phase=='qualification' else c['protocol']['run_order']
 assert complete['passed'] and complete['contract_sha256']==seal and complete['completed_run_ids']==order and complete['total_forwards']==35*len(order)
 assert not (out/'failure.json').exists()
 assert q.read(out/'protection_before.json')==q.read(out/'protection_after.json')==q.old.protection()
 raw=b''.join(struct.pack('<f',(i%257)/256) for i in range(4*352*352));assert hashlib.sha256(raw).hexdigest()==c['input_sha256']
 runmap={x['entry']['config']['run_id']:x for x in c['runs']};summaries=[];identities=[];last_creation=0
 for runid in order:
  p=out/runid;row=runmap[runid];cfg=row['entry']['config'];claim=q.read(p/'identity.json');launch=q.read(p/'launch.json');b=q.read(p/'binding.json');a=q.read(p/'approval.json');payload=q.read(p/'payload_started.json');done=q.read(p/'done.json');result=q.read(p/'result.json');samples=q.read(p/'samples.json');cleanup=q.read(p/'cleanup.json');model=q.read(p/'model_cpu_ready.json');me=b['worker']
  assert me=={k:claim[k] for k in ('pid','creation_ticks')}=={k:a[k] for k in ('pid','creation_ticks')}==payload['identity']==done['identity']==result['worker']
  assert b['launcher']==launch['launcher'] and me['pid']!=b['launcher']['pid'] and me['creation_ticks']>last_creation;last_creation=me['creation_ticks']
  identities.extend([me,b['launcher']]);assert claim['no_site'] and not claim['torch_imported'] and not claim['directml_imported']
  assert b['worker_image']==os.path.normcase(os.path.abspath(c['expected_worker_image'])) and b['retained_actual_worker_handle'] and b['no_payload_before_approval']
  assert b['parent_chain'][0]['pid']==me['pid'] and b['parent_chain'][-1]['pid']==b['launcher']['pid']
  assert a['nonce']==claim['nonce'] and a['plan_sha256']==q.a.digest(p/'plan.json') and payload['approval_sha256']==q.a.digest(p/'approval.json')
  assert launch['suspended_assigned_before_resume'] and launch['kill_on_job_close'] and not launch['breakaway_enabled']
  assert not cleanup['errors'] and cleanup['result']['job_active_processes']==0 and cleanup['result']['all_retained_exited']
  assert cleanup['result']['limits_readback']=={'flags':8712,'active_process_limit':8,'job_memory_bytes':9*2**30}
  assert model['CPU_loaded_state_bytes_verified'] and model['best_sha256']==row['best_sha256'] and model['selected_epoch']==row['selected_epoch'] and model['tensor_evidence_sha256']==q.a.digest(row['tensor_evidence'])
  assert model['parameters']==c['parameter_counts'][cfg['method']] and model['cpu_threads']==model['interop_threads']==1 and not model['training']
  assert len(model['reader_calls'])==1 and model['reader_calls'][0]['sha256']==row['best_sha256'] and not model['reader_calls'][0]['pickle_executed']
  rt=q.read(p/'torch_imported.json');assert rt['torch']==c['expected_torch'] and rt['torch_directml']==c['expected_directml']
  device=q.read(p/'model_device_ready.json');assert device['device']=='privateuseone:1' and device['input_sha256']==c['input_sha256'] and device['input_shape']==[1,4,352,352]
  assert samples[0]['phase']=='before_payload' and all(x['worker']==me for x in samples)
  assert [x['elapsed'] for x in samples]==sorted(x['elapsed'] for x in samples)
  mn=min(x['host']['available_ram_gib'] for x in samples);mc=min(x['host']['available_commit_gib'] for x in samples);mx=max(x['worker_memory']['private_bytes'] for x in samples)/2**30;ws=max(x['worker_memory']['working_set_bytes'] for x in samples)/2**30
  assert mn>=8.5 and mc>=8.5 and mx<=8 and ws<=9
  assert done['forwards']==35 and done['warmups']==5 and done['measurements']==30 and done['training_steps']==0 and not done['test_images_read'] and not done['numerical_comparison'] and done['phase']==phase
  assert not (p/'worker_error.json').exists() and not (p/'controller_failure.json').exists()
  paths=sorted(p.glob('iteration_*.json'));assert len(paths)==35;times=[];previous=device['monotonic_ns']
  for i,file in enumerate(paths):
   observation=q.read(file);v=observation['interval'];assert observation['identity']==me and observation['iteration']==i and observation['kind']==('warmup' if i<5 else 'measured')
   assert v['start_ns']>=previous and v['end_ns']>v['start_ns'] and v['elapsed_ns']==v['end_ns']-v['start_ns'];previous=v['end_ns']
   assert observation['readback_completed'] and observation['finite'] and not observation['gradient_enabled'] and not observation['requires_grad'] and observation['output_bytes']==495616
   assert observation['output_shape']==[1,1,352,352]
   times.append(v['elapsed_ns']/1e6)
  assert q.read(p/'warmup_done.json')['monotonic_ns']>=q.read(paths[4])['interval']['end_ns'] and q.read(p/'warmup_done.json')['monotonic_ns']<=q.read(paths[5])['interval']['start_ns']
  assert done['monotonic_ns']>=previous
  assert q.a.digest(p/'done.json')==result['done_sha256'] and q.a.digest(p/'samples.json')==result['samples_sha256']
  measured=times[5:];median=statistics.median(measured);assert median==quantile(measured,.5)
  summary={'run_id':runid,'method':cfg['method'],'seed':cfg['seed'],'warmup_ms':times[:5],'measured_ms':measured,'median_ms':median,'p25_ms':quantile(measured,.25),'p75_ms':quantile(measured,.75),'minimum_ms':min(measured),'maximum_ms':max(measured),'min_available_ram_gib':mn,'max_private_gib':mx,'max_working_set_gib':ws,'os_peak_working_set_gib':max(x['worker_memory']['os_peak_working_set_bytes'] for x in samples)/2**30,'sample_count':len(samples),'input_sha256':c['input_sha256']}
  summaries.append(summary)
 q.guard.require_idle();snapshot=w.snapshot();states=[]
 for rec in identities:
  if rec['pid'] not in snapshot:state='absent'
  else:
   handle=w.open_process(rec['pid'])
   try:
    actual=w.ident(handle)
    if actual!=rec:state='different_creation'
    else:assert not w.running(handle);state='matching_exited'
   finally:w.close(handle)
  states.append({**rec,'state':state})
 report={'passed':True,'phase':phase,'script_sha256':q.a.digest(__file__),'complete_sha256':q.a.digest(out/'complete.json'),'contract_sha256':seal,'source_count':len(c['sources']),'run_count':len(summaries),'forward_count':len(summaries)*35,'summaries':summaries,'identity_checks':states,'finish_resources':q.resources(),'scope':'qualification resource results only' if phase=='qualification' else 'single-machine synthetic fixed-shape completed-output latency; not GPU kernel latency, VRAM, accuracy, training or universal speed','utc':q.utc()}
 q.save(H/(phase+'_assessment.json'),report)
 if phase=='matrix':
  groups=[]
  for method in c['protocol']['methods']:
   rr=[x for x in summaries if x['method']==method];assert sorted(x['seed'] for x in rr)==sorted(c['protocol']['seeds'])
   values=[x['median_ms'] for x in rr];mean=statistics.mean(values);sd=statistics.stdev(values)
   assert math.isclose(sd,math.sqrt(sum((x-mean)**2 for x in values)/4),rel_tol=1e-12)
   groups.append({'method':method,'n_checkpoints':5,'mean_run_median_ms':mean,'sample_sd_of_run_medians_ms':sd,'min_run_median_ms':min(values),'max_run_median_ms':max(values),'min_available_ram_gib':min(x['min_available_ram_gib'] for x in rr),'max_private_gib':max(x['max_private_gib'] for x in rr),'max_working_set_gib':max(x['max_working_set_gib'] for x in rr)})
  q.save(H/'matrix_summary.json',{'groups':groups,'warmups_preserved':125,'measured_observations_preserved':750,'qualified_observations_excluded_by_design':175,'no_observations_discarded':True,'qualification_not_paper_timing_population':True,'aggregation':'arithmetic mean and sample SD of five per-checkpoint median latencies; within-run repeats not independent training seeds','assessment_sha256':q.a.digest(H/'matrix_assessment.json')});print(groups)
 print({'phase':phase,'passed':True,'runs':len(order),'min_ram_gib':min(x['min_available_ram_gib'] for x in summaries),'max_private_gib':max(x['max_private_gib'] for x in summaries),'live_original_identities':0})
if __name__=='__main__':audit(sys.argv[1])
