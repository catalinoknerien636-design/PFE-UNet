from pathlib import Path
import sys,contextlib,msvcrt,gc,traceback,collections
H=Path(__file__).resolve().parent;R=H.parent
sys.path.insert(0,str(R/'80_pgsnet_module_qualification'));import qualify80 as q

def main():
 c=q.read(H/'contract.json');q.verify(c);q.guard.require_idle();q.resources(10)
 out=H/'meta_v1';out.mkdir(exist_ok=False)
 try:
  with contextlib.ExitStack() as locks:
   for rel in c['locks']:
    f=locks.enter_context((R/rel).open('r+b'));f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
   before=q.old.protection();torch=q.a.require_runtime();torch.set_num_threads(1)
   from torch.utils._python_dispatch import TorchDispatchMode
   from torch.utils._pytree import tree_flatten
   class Ledger(TorchDispatchMode):
    def __init__(self,state,require_meta=False):
     self.rows={};self.refs={};self.ops=[];self.require_meta=require_meta
     for t in state:self.keep(t,'initial_state_or_input')
    def keep(self,t,op):
     if self.require_meta and t.device.type!='meta':raise ValueError('numerical_tensor_in_full_model_profile')
     s=t.untyped_storage();key=s._cdata
     if key not in self.rows:
      self.rows[key]={'bytes':s.nbytes(),'first_shape':list(t.shape),'dtype':str(t.dtype),'first_operator':op};self.refs[key]=s
      return s.nbytes()
     return 0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
     result=func(*args,**(kwargs or {}));new=0
     for t in tree_flatten(result)[0]:
      if isinstance(t,torch.Tensor):new+=self.keep(t,str(func))
     self.ops.append({'operator':str(func),'new_storage_bytes':new});return result
   tests=[]
   for device in ['cpu','meta']:
    x=torch.ones(2,3,device=device);meter=Ledger([x],device=='meta')
    with meter:
     view=x.view(3,2);clone=x.clone();result=clone+1
    assert sum(r['bytes'] for r in meter.rows.values())==72 and len(meter.rows)==3
    tests.append({'device':device,'view_alias_no_double_count':True,'clone_and_add_new_storages':True,'bytes':72})
   q.save(out/'counter_tests.json',{'passed':True,'checks':tests})
   sys.path.insert(0,str(R/'15_pcod_training'));import pcod_models as models
   sys.path.insert(0,str(R/'20_pcod_backend_diagnostics'));import resize_candidate as resize;import resize_factor2 as factor2
   resize.bilinear_half_pixel=factor2.bilinear_half_pixel
   prior={r['method']:r for r in q.read(R/'83_capacity_compute_evidence/analysis_v1/compute_summary.json')};reports=[]
   for method in c['methods']:
    width=41 if method=='wide_direct' else 32
    with torch.device('meta'):model=resize.install_unet_candidate(models.build_model(method,base_channels=width,seed=42)).eval()
    x=torch.empty(1,4,352,352,device='meta',dtype=torch.float32)
    state=list(model.state_dict().values());meter=Ledger(state+[x],True)
    assert all(not m.training for m in model.modules())
    with torch.no_grad(),meter:y=model(x).sigmoid()
    assert tuple(y.shape)==(1,1,352,352) and not y.requires_grad
    params=sum(p.numel() for p in model.parameters());assert params==prior[method]['trainable_parameters']
    rows=list(meter.rows.values());total=sum(r['bytes'] for r in rows);initial=sum(r['bytes'] for r in rows if r['first_operator']=='initial_state_or_input')
    assert total==initial+sum(op['new_storage_bytes'] for op in meter.ops)
    report={'method':method,'parameters':params,'unique_initial_state_and_input_bytes':initial,'unique_operator_result_bytes':total-initial,'total_distinct_logical_storage_bytes':total,'total_distinct_logical_storage_gib':total/2**30,'distinct_storages':len(rows),'operator_calls':len(meter.ops),'output_shape':list(y.shape),'actual_model_numeric_forward':False,'GPU_used':False,'actual_CPU_or_GPU_peak':False,'not_device_workspace_upper_bound':True}
    q.save(out/(method+'.json'),{'summary':report,'storage_rows':rows,'operator_rows':meter.ops});reports.append(report)
    del model,x,y,meter,state;gc.collect();q.resources(8)
   maximum=max(r['total_distinct_logical_storage_gib'] for r in reports)
   assert maximum<=c['meta_resource_gate_total_logical_gib'],('logical_planning_gate',maximum)
   q.verify(c);after=q.old.protection();assert before==after
   q.save(out/'complete.json',{'passed':True,'reports':reports,'max_logical_gib':maximum,'resource_gate_gib':c['meta_resource_gate_total_logical_gib'],'interpretation':'Union of distinct meta storage touched in one eval/no_grad forward plus sigmoid, retaining identities to avoid pointer reuse. Not simultaneous live peak, actual allocation, workspace or driver/cache bound. CPU readback destination is additional and small; backend may materialize/alias differently. Permits only separately guarded one-model pilot, never full benchmark by itself.','before':before,'after':after,'runtime':str(torch.__version__),'resources':q.resources(),'utc':q.utc()})
  print(q.read(out/'complete.json'))
 except BaseException as exc:q.save(out/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'utc':q.utc()});raise
if __name__=='__main__':main()
