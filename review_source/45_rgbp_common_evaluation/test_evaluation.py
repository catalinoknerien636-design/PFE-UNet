import ast
import unittest
import cv2
import numpy as np
import torch
from torch.nn import functional as F
import evaluation as e


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        source = e.ROOT / '12_training/train_rgbp.py'
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Metrics')
        ns = {'torch': torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
        cls.OriginalMetrics = ns['Metrics']

    def sample(self, n=352):
        logits = np.linspace(-3, 3, n*n, dtype=np.float32).reshape(n, n)
        p = torch.sigmoid(torch.from_numpy(logits)).numpy()
        mask = np.zeros((512, 612), np.uint8)
        mask[61:491, 80:550] = 255
        return logits, p, mask

    def test_01_352_no_probability_change(self):
        _, p, m = self.sample()
        q, _ = e.prepare_grid(p, m, raw_grid=(352, 352))
        np.testing.assert_array_equal(p, q)

    def test_02_native_mask_matches_frozen_loader(self):
        _, p, m = self.sample()
        _, y = e.prepare_grid(p, m, raw_grid=(352, 352))
        np.testing.assert_array_equal(y, cv2.resize((m>0).astype(np.float32),(352,352),interpolation=cv2.INTER_NEAREST))

    def test_03_416_resize_half_pixel_oracle(self):
        _, p, m = self.sample(416)
        q, _ = e.prepare_grid(p,m,raw_grid=(416,416))
        oracle = F.interpolate(torch.from_numpy(p[None,None]),(352,352),mode='bilinear',align_corners=False).numpy()[0,0]
        np.testing.assert_allclose(q,oracle,rtol=0,atol=2e-7)

    def test_04_same_native_mask_for_both_methods(self):
        _, p, m = self.sample()
        _, big, _ = self.sample(416)
        _, y1 = e.prepare_grid(p,m,raw_grid=(352,352))
        _, y2 = e.prepare_grid(big,m,raw_grid=(416,416))
        np.testing.assert_array_equal(y1,y2)

    def test_05_dont_threshold_before_resizing(self):
        p = np.full((416,416), .4, np.float32);p[:,::2]=.7
        m = np.zeros((512,612),np.uint8)
        q,_=e.prepare_grid(p,m,raw_grid=(416,416))
        wrong=cv2.resize((p>.5).astype(np.float32),(352,352),interpolation=cv2.INTER_LINEAR)>.5
        self.assertTrue(np.any((q>.5)!=wrong))

    def test_06_exact_frozen_dense_metric_parity(self):
        logits,p,m=self.sample()
        _,y=e.prepare_grid(p,m,raw_grid=(352,352))
        original=self.OriginalMetrics()
        original.update(torch.from_numpy(logits[None,None]),torch.from_numpy(y[None,None]),['a'])
        actual=e.aggregate([e.image_metrics('a',p,m,raw_grid=(352,352))])
        for k,v in original.finalize().items():self.assertEqual(actual[k],v,k)

    def test_07_exact_batch_split_metric_parity(self):
        logits,p,m=self.sample()
        rows=[];original=self.OriginalMetrics()
        for stem,lo,pr in [('a',logits,p),('b',-logits,torch.sigmoid(torch.from_numpy(-logits)).numpy())]:
            _,y=e.prepare_grid(pr,m,raw_grid=(352,352))
            original.update(torch.from_numpy(lo[None,None]),torch.from_numpy(y[None,None]),[stem])
            rows.append(e.image_metrics(stem,pr,m,raw_grid=(352,352)))
        actual=e.aggregate(rows)
        for k,v in original.finalize().items():self.assertEqual(actual[k],v,k)

    def test_08_boundary_reuses_original_definition(self):
        _,p,m=self.sample();q,y=e.prepare_grid(p,m,raw_grid=(352,352))
        expected=e.boundary.pair_metrics(q>.5,y>.5)
        actual=e.image_metrics('a',p,m,raw_grid=(352,352))
        for k,v in expected.items():self.assertEqual(actual[k],v)

    def test_09_empty_dense_and_boundary_conventions_differ(self):
        r=e.image_metrics('a',np.zeros((352,352),np.float32),np.zeros((4,5),np.uint8),raw_grid=(352,352))
        result=e.aggregate([r])
        self.assertEqual(result['iou'],0)
        self.assertEqual(result['boundary_f1_2px_image_mean'],1)
        self.assertEqual(result['boundary_iou_pooled'],1)

    def test_10_one_empty_boundary_zero(self):
        r=e.image_metrics('a',np.zeros((352,352),np.float32),np.ones((4,5),np.uint8),raw_grid=(352,352))
        self.assertEqual(r['boundary_f1_2px'],0)

    def test_11_threshold_half_is_background(self):
        r=e.image_metrics('a',np.full((352,352),.5,np.float32),np.ones((4,5),np.uint8),raw_grid=(352,352))
        self.assertEqual(r['tp'],0)

    def test_12_no_uint8_prediction(self):
        with self.assertRaises(ValueError):e.prepare_grid(np.zeros((352,352),np.uint8),np.zeros((4,5),np.uint8),raw_grid=(352,352))

    def test_13_no_float64_prediction(self):
        with self.assertRaises(ValueError):e.prepare_grid(np.zeros((352,352),np.float64),np.zeros((4,5),np.uint8),raw_grid=(352,352))

    def test_14_no_logits_nan_or_inf(self):
        for value in (-.1,1.1,np.nan,np.inf):
            with self.assertRaises(ValueError):e.prepare_grid(np.full((352,352),value,np.float32),np.zeros((4,5),np.uint8),raw_grid=(352,352))

    def test_15_declared_grid_must_match(self):
        with self.assertRaises(ValueError):e.prepare_grid(np.zeros((352,352),np.float32),np.zeros((4,5),np.uint8),raw_grid=(416,416))

    def test_16_no_other_output_grid(self):
        with self.assertRaises(ValueError):e.prepare_grid(np.zeros((512,512),np.float32),np.zeros((4,5),np.uint8),raw_grid=(512,512))

    def test_17_invalid_mask_no_implicit_repair(self):
        for m in (np.full((4,5),127,np.uint8),np.zeros((4,5),np.float32),np.zeros((4,5,3),np.uint8)):
            with self.assertRaises(ValueError):e.prepare_grid(np.zeros((352,352),np.float32),m,raw_grid=(352,352))

    def test_18_full_report_and_fixed_strata(self):
        _,p,m=self.sample();rows=[e.image_metrics(k,p,m,raw_grid=(352,352)) for k in ('a','b','c')]
        result=e.complete_report(rows,['a','b','c'],{'a':'train','b':'validation','c':'none_detected'})
        self.assertEqual(result['all_test']['images'],3)
        self.assertTrue(all(x['images']==1 for x in result['association_strata'].values()))

    def test_19_reject_subset_and_reordering(self):
        _,p,m=self.sample();r=e.image_metrics('a',p,m,raw_grid=(352,352))
        for rows in ([r],[r,r]):
            with self.assertRaises(ValueError):e.complete_report(rows,['a','b'],{'a':'train','b':'validation'})

    def test_20_reject_changed_strata(self):
        _,p,m=self.sample();r=e.image_metrics('a',p,m,raw_grid=(352,352))
        with self.assertRaises(ValueError):e.complete_report([r],['a'],{'a':'independent'})

    def test_21_reject_count_corruption(self):
        _,p,m=self.sample();r=e.image_metrics('a',p,m,raw_grid=(352,352));r['tp']+=1
        with self.assertRaises(ValueError):e.aggregate([r])

    def test_22_three_seed_smallest_nonmid_p(self):
        r=e.paired_three_seed({s:.1 for s in e.SEEDS},{s:0 for s in e.SEEDS})
        self.assertEqual(r['two_sided_non_mid_p'],.25)
        self.assertAlmostEqual(r['sample_sd'],0,places=15)

    def test_23_zero_differences_p_one(self):
        self.assertEqual(e.paired_three_seed(dict.fromkeys(e.SEEDS,0),dict.fromkeys(e.SEEDS,0))['two_sided_non_mid_p'],1)

    def test_24_paired_ids_not_3_vs_5(self):
        with self.assertRaises(ValueError):e.paired_three_seed({42:1,2026:1,3407:1,7819:1},dict.fromkeys(e.SEEDS,0))

    def test_25_paired_order_not_dict_insertion(self):
        r=e.paired_three_seed({3407:3,42:1,2026:2},{2026:1,42:0,3407:1})
        self.assertEqual(r['pfe_minus_direct'],[1,1,2])
        self.assertAlmostEqual(r['sample_sd'],np.std([1,1,2],ddof=1))

    def test_26_holm_two_prespecified_comparisons(self):
        p=dict(zip(e.VARIANTS,(.25,.5)))
        self.assertEqual(e.holm_two_primary(p),dict.fromkeys(e.VARIANTS,.5))

    def test_27_holm_rejects_secondary_family(self):
        with self.assertRaises(ValueError):e.holm_two_primary({'film':.1,'rgb':.2})

    def test_28_holm_capped_and_monotone(self):
        p=dict(zip(e.VARIANTS,(.75,1.)))
        self.assertEqual(e.holm_two_primary(p),dict.fromkeys(e.VARIANTS,1.))
