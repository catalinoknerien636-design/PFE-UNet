"""Small CPU-only execution boundary and frozen metric checks; no real data/models."""
import unittest,tempfile,copy,json
from pathlib import Path
import run73_v2 as a
class Tiny(a.torch.nn.Module):
 def __init__(self,nonfinite=False):super().__init__();self.nonfinite=nonfinite
 def forward(self,x):
  y=x[:,:1]*0
  if self.nonfinite:y[:]=float('nan')
  return y
class Checks(unittest.TestCase):
 def test_complete_population(self):a.exact_population([{'sample_id':'a'},{'sample_id':'b'}],['a','b'])
 def test_reordered(self):
  with self.assertRaises(ValueError):a.exact_population([{'sample_id':'b'},{'sample_id':'a'}],['a','b'])
 def test_missing(self):
  with self.assertRaises(ValueError):a.exact_population([{'sample_id':'a'}],['a','b'])
 def test_duplicate(self):
  with self.assertRaises(ValueError):a.exact_population([{'sample_id':'a'},{'sample_id':'a'}],['a','a'])
 def test_all58_contract(self):a.contract_check(a.read(a.HERE/'contract_v2.json'))
 def test_remove_run_rejected(self):
  c=a.read(a.HERE/'contract_v2.json');c['runs'].pop()
  with self.assertRaises(ValueError):a.contract_check(c)
 def test_reorder_runs_rejected(self):
  c=a.read(a.HERE/'contract_v2.json');c['runs'][:2]=c['runs'][1::-1]
  with self.assertRaises(ValueError):a.contract_check(c)
 def test_statistics_change_rejected(self):
  c=a.read(a.HERE/'contract_v2.json');c['statistics']['test']='mid p'
  with self.assertRaises(ValueError):a.contract_check(c)
 def test_failure_relabel_rejected(self):
  c=a.read(a.HERE/'contract_v2.json');c['amendment']['original_failures_relabelled']=True
  with self.assertRaises(ValueError):a.contract_check(c)
 def test_incomplete_summary_rejected(self):
  with self.assertRaises(ValueError):a.summarize([])
 def test_resource_guard_raises(self):
  original=a.x.q.require_resources
  def fail():raise RuntimeError('resource')
  try:
   a.x.q.require_resources=fail
   with self.assertRaises(RuntimeError):list(a.GuardedLoader([{'x':1}]))
  finally:a.x.q.require_resources=original
 def export(self,family,ids=('11111111111111','22222222222222'),nonfinite=False,loader_ids=None,fail_commit=False):
  ep=a.pcod.e if family=='PCOD' else a.rgbp.e;channels=4 if family=='PCOD' else 6
  with tempfile.TemporaryDirectory(dir=a.HERE/'cpu_tests_v2') as td:
   out=Path(td)/'export';prov={'contract_sha256':'test','export_contract_sha256':'test'}
   loader=[{'image':a.torch.zeros(2,channels,352,352),'stem':list(loader_ids or ids)}]
   def check():
    if fail_commit:raise ValueError('commit rejected')
   try:
    if family=='PCOD':ep.export_batches(Tiny(nonfinite),loader,'cpu',out,list(ids),prov,check)
    else:ep.export_batches(Tiny(nonfinite),loader,'cpu',out,list(ids),[352,352],prov,check)
   except ValueError:
    self.assertTrue((out/'failure.json').exists());self.assertFalse((out/'export_complete.json').exists());raise
   b=ep.verify_receipt(out/'bundle.json','test',list(ids));self.assertEqual(len(b['items']),2)
   v=a.pcod.e.read_probability(out/b['items'][0]['path'],b['items'][0]['sha256'],[352,352]);self.assertTrue((v==.5).all())
 def test_pcod_export_exact(self):self.export('PCOD')
 def test_rgbp_export_exact(self):self.export('RGBP')
 def test_nonfinite_pcod(self):
  with self.assertRaises(ValueError):self.export('PCOD',nonfinite=True)
 def test_nonfinite_rgbp(self):
  with self.assertRaises(ValueError):self.export('RGBP',nonfinite=True)
 def test_missing_export_pcod(self):
  with self.assertRaises(ValueError):self.export('PCOD',ids=('11111111111111','22222222222222','33333333333333'),loader_ids=('11111111111111','22222222222222'))
 def test_reordered_export_rgbp(self):
  with self.assertRaises(ValueError):self.export('RGBP',loader_ids=('22222222222222','11111111111111'))
 def test_commit_failure_no_receipt(self):
  with self.assertRaises(ValueError):self.export('PCOD',fail_commit=True)
 def test_same_fixed_metrics(self):
  n=a.np;y=n.zeros((352,352),n.float32);y[5:9,5:9]=1;p=y.copy();p[8,8]=.5
  left=a.pcod.e.image_metrics('a',p,y);right=a.pcod.e.metrics.image_metrics('a',p,(y*255).astype(n.uint8),raw_grid=[352,352]);self.assertEqual(left,right);self.assertEqual(left['tp'],15);self.assertEqual(left['fn'],1)
 def test_empty_rules(self):
  p=a.np.zeros((352,352),a.np.float32);row=a.pcod.e.image_metrics('a',p,p);r=a.pcod.e.aggregate([row]);self.assertEqual(r['iou'],0);self.assertEqual(r['boundary_iou_pooled'],1)
 def test_p_minima(self):
  e=a.pcod.e;five=e.paired({s:.6 for s in e.SEEDS},{s:.5 for s in e.SEEDS},'direct_reference',sign_flip=True);three=e.metrics.paired_three_seed({s:.6 for s in e.SEEDS[:3]},{s:.5 for s in e.SEEDS[:3]});self.assertEqual(five['two_sided_non_mid_p'],.0625);self.assertEqual(three['two_sided_non_mid_p'],.25)
 def test_holm(self):
  ev=a.pcod.e.metrics;r=ev.holm_two_primary(dict(zip(ev.VARIANTS,[.25,.5])));self.assertEqual(sorted(r.values()),[.5,.5])
 def test_member_same_count_replacement_rejected(self):
  c=a.read(a.HERE/'contract_v2.json');c['members']['RGBP'][0]['image']='test/image/99999999999999_rgb.tiff'
  with self.assertRaises(ValueError):a.contract_check(c)
 def test_missing_rgbp_input_binding_rejected(self):
  c=a.read(a.HERE/'contract_v2.json');del c['members']['RGBP'][0]['image_sha256']
  with self.assertRaises(ValueError):a.contract_check(c)
 def test_contract_changed_mid_run_rejected(self):
  original,seal=a.CONTRACT,a._sealed_sha
  try:
   with tempfile.TemporaryDirectory(dir=a.HERE/'cpu_tests_v2') as td:
    a.CONTRACT=Path(td)/'c.json';a.CONTRACT.write_text('{"value":1}');a.load_contract();first=a.contract_sha();a.CONTRACT.write_text('{"value":2}')
    with self.assertRaises(ValueError):a.contract_sha()
  finally:a.CONTRACT,a._sealed_sha=original,seal
 def test_exact_loader_compare_rejects_changed_tensor(self):
  for fn in [a.pcod.core.backend.h.compare_tree,a.x.c.backend.compare_tree]:
   with self.assertRaises(AssertionError):fn({'a':a.torch.tensor([0.])},{'a':a.torch.tensor([1.])},{'atol':0,'rtol':0})
if __name__=='__main__':
 a.torch.set_num_threads(1);out=a.HERE/'cpu_tests_v2';out.mkdir(exist_ok=False)
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
 a.save(out/'report.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,'failures':[(str(t),s) for t,s in result.failures+result.errors],'runner_sha256':a.sha(a.HERE/'run73_v2.py'),'test_sha256':a.sha(__file__),'contract_sha256':a.sha(a.HERE/'contract_v2.json'),'GPU_used':False,'test_pixels_read':False,'real_checkpoint_loaded':False})
 raise SystemExit(0 if result.wasSuccessful() else 1)
