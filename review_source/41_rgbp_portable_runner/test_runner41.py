import copy
from contextlib import redirect_stdout
import io
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
import rgbp_core as c
import rgbp_protocol as p
queue_spec=importlib.util.spec_from_file_location('rgbp_queue41',c.HERE/'run_queue.py')
q=importlib.util.module_from_spec(queue_spec)
queue_spec.loader.exec_module(q)

ATTEMPT = None
FIXTURES = {}


class TinyDataset:
    def __init__(self,channels,count,part):
        generator = c.t.torch.Generator().manual_seed(811 if part == 'train' else 912)
        self.items = [{'image':c.t.torch.rand(channels,32,32,generator=generator),
            'mask':(c.t.torch.rand(1,32,32,generator=generator)>.5).float(),
            'stem':f'{part}{i}'} for i in range(count)]
    def __len__(self):return len(self.items)
    def __getitem__(self,index):return self.items[index]


def build_small_fixtures():
    for method in c.candidate.METHODS:
        base = dict(c.configurations(True)[0],method=method,purpose='hardware_smoke',
            device='cpu',expected_device_name='CPU',image_size=32,base_channels=4,
            epochs=2,batch_size=2,sample_counts={'train':4,'validation':3},
            sample_limits={'train':4,'validation':3},run_id='synthetic_'+method)
        full = dict(base,output_dir=str(ATTEMPT/'models'/method/'full'))
        split = dict(base,output_dir=str(ATTEMPT/'models'/method/'split'))
        def fixture_identity(cfg):
            return {'kind':'cpu_fixture_not_formal','config_sha256':c.t.object_sha(cfg)}
        def dataset(root,manifest,part,**kwargs):
            return TinyDataset(3 if method == 'rgb_only' else 6,4 if part=='train' else 3,part)
        ids = ['validation0','validation1','validation2']
        with mock.patch.object(c,'identity',side_effect=fixture_identity), \
                mock.patch.object(c,'validation_ids',return_value=ids), \
                mock.patch.object(c.t,'RGBPGlassDataset',side_effect=dataset), redirect_stdout(io.StringIO()):
            c.execute(full,fixture_identity(full))
            with c.installed_runtime():
                c.t.run_config(split,stop_after_epoch=1)
            with unittest.TestCase().assertRaisesRegex(ValueError,'explicit'):
                c.execute(split,fixture_identity(split))
            c.execute(split,fixture_identity(split),allow_resume=True)
        a = c.load_checkpoint(full,fixture_identity(full),ids)
        b = c.load_checkpoint(split,fixture_identity(split),ids)
        for name in ('model_state','optimizer_state','best_model_state'):
            c.backend.compare_tree(a[name],b[name],{'atol':0,'rtol':0})
        assert c.t.torch.equal(a['rng']['torch_cpu'],b['rng']['torch_cpu'])
        assert a['rng']['python'] == b['rng']['python']
        assert a['rng']['numpy'][0] == b['rng']['numpy'][0]
        c.t.np.testing.assert_array_equal(a['rng']['numpy'][1],b['rng']['numpy'][1])
        assert a['rng']['numpy'][2:] == b['rng']['numpy'][2:]
        assert a['best_validation_rows'] == b['best_validation_rows']
        assert [{k:v for k,v in r.items() if not k.endswith('_seconds')} for r in a['history']] == [
            {k:v for k,v in r.items() if not k.endswith('_seconds')} for r in b['history']]
        c.verify_exports(split,fixture_identity(split),ids)
        FIXTURES[method] = (split,fixture_identity(split),ids)


class AdmissionTests(unittest.TestCase):
    def test_runner_module_is_local_41(self):
        self.assertEqual(Path(q.__file__).resolve(),(c.HERE/'run_queue.py').resolve())
    def test_30_disabled_candidates(self):
        configs=c.configurations(False)
        self.assertEqual(len(configs),30)
        self.assertFalse(any(x['training_authorized'] or x['protocol_frozen'] for x in configs))
    def test_disabled_identity_rejected(self):
        with self.assertRaises(ValueError):c.identity(c.configurations(False)[0])
    def test_changed_epoch_rejected(self):
        cfg=c.configurations(True)[0];cfg['epochs']=49
        with self.assertRaises(ValueError):c.identity(cfg)
    def test_seed17_rejected(self):
        cfg=c.configurations(True)[0];cfg['seed']=17
        with self.assertRaises(ValueError):c.identity(cfg)
    def test_old_output_rejected(self):
        cfg=c.configurations(True)[0];cfg['output_dir']=str(c.ROOT/'12_training/runs/old')
        with self.assertRaises(ValueError):c.identity(cfg)
    def test_identity_has_no_device_access(self):
        with mock.patch.object(c.t,'choose_device',side_effect=AssertionError('device')):
            value=c.identity(c.configurations(True)[0])
        self.assertEqual(value['execution_engine'],'rgbp_portable41_v1')
    def test_disabled_protocol_rejected(self):
        with self.assertRaises(ValueError):p.validate({'execution_enabled':False})
    def test_missing_review_blocks_freeze_without_release(self):
        target=ATTEMPT/'absent_protocol.json'
        with mock.patch.object(p,'PROTOCOL',target),mock.patch.object(p,'REVIEW',ATTEMPT/'absent_review.json'):
            with self.assertRaises(FileNotFoundError):p.freeze()
        self.assertFalse(target.exists())
    def test_incomplete_science_review_rejected(self):
        with self.assertRaises(ValueError):p.validate_review({'checks':{'epochs':50},'rationale':'fixture'})
    def test_review_missing_sources_rejected(self):
        with self.assertRaises(ValueError):p.validate_review({'checks':p.REVIEW_CHECKS,'rationale':'fixture'})
    def test_claiming_physical_independence_rejected(self):
        checks=dict(p.REVIEW_CHECKS,physical_scene_independence_still_unresolved=False)
        with self.assertRaises(ValueError):p.validate_review({'checks':checks,'rationale':'fixture'})
    def test_queue_refuses_before_lock_or_device(self):
        with mock.patch.object(p,'read',return_value={'execution_enabled':False}), \
                mock.patch.object(q,'byte_lock') as lock,mock.patch.object(c.t,'choose_device') as device:
            with self.assertRaises(ValueError):q.run()
        lock.assert_not_called();device.assert_not_called()
    def test_low_disk_rejected(self):
        with mock.patch.object(q.shutil,'disk_usage',return_value=SimpleNamespace(free=19*1024**3)):
            with self.assertRaises(RuntimeError):q.require_resources()
    def test_low_memory_rejected(self):
        with mock.patch.object(q.shutil,'disk_usage',return_value=SimpleNamespace(free=30*1024**3)), \
                mock.patch.object(q,'available_memory',return_value=7*1024**3):
            with self.assertRaises(RuntimeError):q.require_resources()
    def test_resource_boundary(self):
        with mock.patch.object(q.shutil,'disk_usage',return_value=SimpleNamespace(free=20*1024**3)), \
                mock.patch.object(q,'available_memory',return_value=8*1024**3):q.require_resources()
    def test_runtime_restored_after_exception(self):
        original=(c.t.build_model,c.t.validate_config,c.backend.resize.bilinear_half_pixel)
        with self.assertRaises(RuntimeError):
            with c.installed_runtime():raise RuntimeError('fixture')
        self.assertEqual(original,(c.t.build_model,c.t.validate_config,c.backend.resize.bilinear_half_pixel))
    def test_real_byte_lock_excludes_other_process(self):
        path=ATTEMPT/'fixture.lock'
        code="import msvcrt,sys; f=open(sys.argv[1],'r+b'); f.seek(0);\ntry: msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)\nexcept OSError: sys.exit(0)\nelse: sys.exit(9)"
        with q.byte_lock(path,create=True):
            result=subprocess.run([sys.executable,'-B','-c',code,str(path)],capture_output=True,
                timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
            self.assertEqual(result.returncode,0,result.stderr)
        with q.byte_lock(path):pass
    def test_own_ancestry_only_excluded_from_process_inventory(self):
        rows=[{'ProcessId':1,'ParentProcessId':0,'CommandLine':'python project'},
              {'ProcessId':2,'ParentProcessId':1,'CommandLine':'python pfe_unet_research_extension_20260923'},
              {'ProcessId':3,'ParentProcessId':0,'CommandLine':'python pfe_unet_research_extension_20260923'}]
        self.assertEqual([r['ProcessId'] for r in q.guard.other_project_processes(rows,2)],[3])
    def test_nonfinite_tree_rejected(self):
        with self.assertRaises(ValueError):c.finite_tree({'state':[c.t.torch.tensor(float('nan'))]})
    def test_sidecar_before_deserialization(self):
        cfg=dict(c.configurations(True)[0],output_dir=str(ATTEMPT/'foreign'))
        Path(cfg['output_dir']).mkdir()
        c.candidate.save_new(Path(cfg['output_dir'])/'config.json',{'wrong':'sidecar'})
        with mock.patch.object(c.t.torch,'load') as load:
            with self.assertRaises(ValueError):c.inspect_run(cfg,{})
        load.assert_not_called()


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):build_small_fixtures()

    def local_fixture(self):
        cfg,identity,ids=FIXTURES['direct']
        out=ATTEMPT/self._testMethodName
        shutil.copytree(cfg['output_dir'],out)
        changed=dict(cfg,output_dir=str(out))
        # Explicit local fixture relocation, never a formal identity override.
        c.t.write_json(out/'config.json',changed)
        return out,changed,identity,ids

    def test_five_methods_exact_recovery(self):self.assertEqual(set(FIXTURES),set(c.candidate.METHODS))
    def test_complete_marker_and_no_overwrite(self):
        _,cfg,identity,ids=self.local_fixture()
        self.assertFalse(c.completed_run(cfg,identity,'fixture',ids))
        c.mark_completed(cfg,identity,'fixture',ids)
        self.assertTrue(c.completed_run(cfg,identity,'fixture',ids))
        with self.assertRaises(FileExistsError):c.mark_completed(cfg,identity,'fixture',ids)
    def test_wrong_protocol_marker(self):
        _,cfg,identity,ids=self.local_fixture();c.mark_completed(cfg,identity,'fixture',ids)
        with self.assertRaises(ValueError):c.completed_run(cfg,identity,'other',ids)
    def test_wrong_status_hash(self):
        out,cfg,identity,ids=self.local_fixture();record=c.read(out/'status.json');record['checkpoint_sha256']='wrong'
        c.t.write_json(out/'status.json',record)
        with self.assertRaises(ValueError):c.verify_exports(cfg,identity,ids)
    def test_test_evaluated_flag(self):
        out,cfg,identity,ids=self.local_fixture();record=c.read(out/'status.json');record['test_evaluated']=True
        c.t.write_json(out/'status.json',record)
        with self.assertRaises(ValueError):c.verify_exports(cfg,identity,ids)
    def test_wrong_best_epoch(self):
        out,cfg,identity,ids=self.local_fixture();best=c.t.torch.load(out/'best.pt',weights_only=False)
        best['epoch']=99;c.t.save_checkpoint(out/'best.pt',best)
        with self.assertRaises(ValueError):c.verify_exports(cfg,identity,ids)
    def test_changed_best_weight(self):
        out,cfg,identity,ids=self.local_fixture();best=c.t.torch.load(out/'best.pt',weights_only=False)
        name=next(k for k,v in best['model_state'].items() if v.is_floating_point())
        best['model_state'][name]+=1;c.t.save_checkpoint(out/'best.pt',best)
        with self.assertRaises(AssertionError):c.verify_exports(cfg,identity,ids)
    def test_changed_per_image_export(self):
        out,cfg,identity,ids=self.local_fixture();rows=c.read(out/'best_validation_per_image.json');rows[0]['mae']=.999
        c.t.write_json(out/'best_validation_per_image.json',rows)
        with self.assertRaises(ValueError):c.verify_exports(cfg,identity,ids)
    def test_changed_history_export(self):
        out,cfg,identity,ids=self.local_fixture()
        with (out/'history.csv').open('a',encoding='utf-8') as stream:stream.write('invalid\n')
        with self.assertRaises(ValueError):c.verify_exports(cfg,identity,ids)
    def test_marker_cannot_hide_changed_artifact(self):
        out,cfg,identity,ids=self.local_fixture();c.mark_completed(cfg,identity,'fixture',ids)
        record=c.read(out/'status.json');record['extra']='changed';c.t.write_json(out/'status.json',record)
        with self.assertRaises(ValueError):c.completed_run(cfg,identity,'fixture',ids)
    def test_nonfull_budget_rejected(self):
        _,cfg,identity,ids=self.local_fixture();cfg=dict(cfg,epochs=3)
        with self.assertRaises(ValueError):c.verify_exports(cfg,identity,ids)


if __name__=='__main__':
    raise SystemExit('Use run_checks.py for exclusive attempt directories')
