from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import unittest
from unittest import mock
import formal_contract as f


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg=copy.deepcopy(f.candidates()[0])
        self.fixture_cfg=dict(self.cfg,epochs=2,image_size=32,sample_counts={'train':4,'validation':3})
        self.ids=['v0','v1','v2']
        self.identity={'kind':'synthetic_contract_fixture_not_result','seed':42}
        best_rows=[{'stem':sid,'tp':1,'fp':1,'fn':1,'tn':1021,'mae':.25} for sid in self.ids]
        row={'train_images':4,'val_images':3,'train_loss':1.,'val_loss':1.,'val_iou':3/(9+1e-7),
            'val_f1':6/(12+1e-7),'val_image_mae':.25,'val_tp':3,'val_fp':3,'val_fn':3,'val_tn':3063}
        self.cp={'identity':self.identity,'epoch':2,'history':[dict(row,epoch=i) for i in (1,2)],
            'best_epoch':1,'best_iou':row['val_iou'],'model_state':{'fixture':1},
            'best_model_state':{'fixture':1},'optimizer_state':{'fixture':1},
            'rng':{'torch_cpu':1,'python':1,'numpy':1},'best_validation_rows':best_rows}

    def validate(self):
        return f.validate_checkpoint(self.cp,self.fixture_cfg,self.identity,self.ids)

    def test_matrix_is_30_disabled(self):
        rows=f.candidates()
        self.assertEqual(len(rows),30)
        self.assertFalse(any(r['training_authorized'] or r['protocol_frozen'] for r in rows))
        self.assertEqual({r['seed'] for r in rows},{42,2026,3407})

    def test_dropped_method_rejected(self):
        with self.assertRaises(ValueError): f.validate_matrix(f.candidates()[:-1])

    def test_duplicate_rejected(self):
        rows=f.candidates();rows[-1]=rows[0]
        with self.assertRaises(ValueError):f.validate_matrix(rows)

    def test_development_seed_rejected(self):
        self.cfg['seed']=17
        with self.assertRaises(ValueError):f.candidate_identity(self.cfg)

    def test_budget_change_rejected(self):
        self.cfg['epochs']=49
        with self.assertRaises(ValueError):f.candidate_identity(self.cfg)

    def test_activation_rejected(self):
        self.cfg.update(training_authorized=True,protocol_frozen=True,candidate_only=False)
        with self.assertRaises(ValueError):f.candidate_identity(self.cfg)

    def test_history_warmstart_rejected(self):
        self.cfg['initialization']='development_checkpoint'
        with self.assertRaises(ValueError):f.candidate_identity(self.cfg)

    def test_candidate_does_not_select_gpu(self):
        with mock.patch.object(f.t,'choose_device',side_effect=AssertionError('GPU access')):
            value=f.candidate_identity(self.cfg)
        self.assertFalse(value['executable_formal_identity'])

    def test_test_retention(self):
        with self.assertRaises(ValueError):f.validate_members({'a':'test','b':'train'},{'a':'quarantine','b':'train'})

    def test_train_to_val_reassignment_rejected(self):
        with self.assertRaises(ValueError):f.validate_members({'a':'train'},{'a':'validation'})

    def test_explicit_quarantine(self):
        f.validate_members({'a':'train','b':'test'},{'a':'quarantine','b':'test'})

    def test_correct_checkpoint(self):
        self.assertEqual(self.validate(),2)

    def test_development_checkpoint_rejected(self):
        self.cp['identity']={'seed':17}
        with self.assertRaises(ValueError):self.validate()

    def test_partial_coverage(self):
        self.cp['history'][1]['train_images']=3
        with self.assertRaises(ValueError):self.validate()

    def test_epoch_gap(self):
        self.cp['history'][1]['epoch']=3
        with self.assertRaises(ValueError):self.validate()

    def test_budget_overrun(self):
        self.cp['epoch']=3
        with self.assertRaises(ValueError):self.validate()

    def test_best_exact_tie_must_be_first(self):
        self.cp['best_epoch']=2
        with self.assertRaises(ValueError):self.validate()

    def test_nonfinite_metric(self):
        self.cp['history'][1]['val_iou']=float('nan')
        with self.assertRaises(ValueError):self.validate()

    def test_missing_rng(self):
        del self.cp['rng']['numpy']
        with self.assertRaises(ValueError):self.validate()

    def test_foreign_validation_ids(self):
        self.cp['best_validation_rows'][1]['stem']='unknown'
        with self.assertRaises(ValueError):self.validate()

    def test_duplicate_validation_ids(self):
        self.cp['best_validation_rows'][1]['stem']='v0'
        with self.assertRaises(ValueError):self.validate()

    def test_wrong_image_counts(self):
        self.cp['best_validation_rows'][1]['tn']+=1
        with self.assertRaises(ValueError):self.validate()

    def test_changed_mae_aggregation(self):
        self.cp['best_validation_rows'][1]['mae']=.5
        with self.assertRaises(ValueError):self.validate()

    def test_changed_f1_aggregation(self):
        self.cp['history'][0]['val_f1']=.4
        with self.assertRaises(ValueError):self.validate()

    def test_sidecars_rejected_before_load(self):
        path=f.HERE/'synthetic_sidecar_rejection_fixture';path.mkdir()
        f.save_new(path/'config.json',{'wrong':'metadata'})
        cfg=dict(self.fixture_cfg,output_dir=str(path))
        loader=mock.Mock(side_effect=AssertionError('Must not deserialize'))
        with self.assertRaises(ValueError):f.inspect_resume(cfg,self.identity,self.ids,loader)
        loader.assert_not_called()

    def test_source_runtime_restored_on_error(self):
        original=f.t.build_model;resize=f.backend.resize.bilinear_half_pixel
        with self.assertRaises(RuntimeError):
            with f.corrected_cpu_test_runtime():
                self.assertIs(f.t.build_model,f.backend.corrected_model)
                raise RuntimeError('Synthetic context rollback')
        self.assertIs(f.t.build_model,original)
        self.assertIs(f.backend.resize.bilinear_half_pixel,resize)


class TinySyntheticDataset:
    def __init__(self,channels,count,prefix):
        generator=f.t.torch.Generator().manual_seed(8040 if prefix=='train' else 9040)
        self.items=[{'image':f.t.torch.rand(channels,32,32,generator=generator),
            'mask':(f.t.torch.rand(1,32,32,generator=generator)>.5).float(),
            'stem':f'{prefix}{i}'} for i in range(count)]

    def __len__(self):return len(self.items)
    def __getitem__(self,index):return self.items[index]


class ResumeTests(unittest.TestCase):
    def test_five_small_models_cpu_resume(self):
        reports=[]
        for method in f.METHODS:
            with self.subTest(method=method):
                root=f.HERE/'cpu_resume_fixture_v1'/method
                self.assertFalse(root.exists())
                train=TinySyntheticDataset(3 if method=='rgb_only' else 6,4,'train')
                val=TinySyntheticDataset(3 if method=='rgb_only' else 6,3,'validation')
                cfg=dict(f.candidates()[0],method=method,run_id='cpu_fixture_'+method,purpose='hardware_smoke',
                    device='cpu',expected_device_name='CPU',image_size=32,base_channels=4,batch_size=2,
                    epochs=2,sample_counts={'train':4,'validation':3},sample_limits={'train':4,'validation':3})
                identity={'kind':'small_cpu_recovery_fixture_not_formal','method':method}
                full=dict(cfg,output_dir=str(root/'full'))
                split=dict(cfg,output_dir=str(root/'split'))
                with f.corrected_cpu_test_runtime(),redirect_stdout(io.StringIO()):
                    f.t.fit(full,train,val,identity)
                    f.t.fit(split,train,val,identity,stop_after_epoch=1)
                    state=f.inspect_resume(split,identity,[r['stem'] for r in val.items])
                    self.assertEqual(state,{'resume':True,'epoch':1})
                    f.t.fit(split,train,val,identity,resume=True)
                left=f.t.torch.load(root/'full/latest.pt',map_location='cpu',weights_only=False)
                right=f.t.torch.load(root/'split/latest.pt',map_location='cpu',weights_only=False)
                for name in ('model_state','optimizer_state','best_model_state'):
                    f.backend.compare_tree(left[name],right[name],{'atol':0,'rtol':0})
                self.assertTrue(f.t.torch.equal(left['rng']['torch_cpu'],right['rng']['torch_cpu']))
                self.assertEqual(left['rng']['python'],right['rng']['python'])
                self.assertEqual(left['rng']['numpy'][0],right['rng']['numpy'][0])
                f.t.np.testing.assert_array_equal(left['rng']['numpy'][1],right['rng']['numpy'][1])
                self.assertEqual(left['rng']['numpy'][2:],right['rng']['numpy'][2:])
                self.assertEqual(left['best_validation_rows'],right['best_validation_rows'])
                self.assertEqual(f.validate_checkpoint(right,split,identity,[r['stem'] for r in val.items]),2)
                for a,b in zip(left['history'],right['history']):
                    self.assertEqual({k:v for k,v in a.items() if not k.endswith('_seconds')},
                        {k:v for k,v in b.items() if not k.endswith('_seconds')})
                reports.append({'method':method,'cpu':True,'width':4,'size':32,'epochs':2,
                    'exact_model_optimizer_rng_metrics_resume':True,'formal_result':False})
        f.save_new(f.HERE/'small_cpu_resume_checks.json',reports)


if __name__=='__main__':unittest.main()
