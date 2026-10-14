import hashlib
from pathlib import Path
import unittest
import numpy as np
import offline_evaluate as o

FIXTURES = Path(__file__).resolve().parent/'cpu_attempt_v1/fixtures'


class OfflineChecks(unittest.TestCase):
    def file(self,name,array,extra=b''):
        FIXTURES.mkdir(parents=True,exist_ok=True)
        path=FIXTURES/(name+'.npy')
        with path.open('xb') as f:np.save(f,array,allow_pickle=False);f.write(extra)
        return path,o.sha(path)

    def test_01_exact_npy(self):
        x=np.full((352,352),.5,np.float32);path,digest=self.file('exact',x)
        np.testing.assert_array_equal(o.read_probability(path,digest,(352,352)),x)

    def test_02_reject_hash_change(self):
        path,_=self.file('hash',np.zeros((352,352),np.float32))
        with self.assertRaises(ValueError):o.read_probability(path,'0'*64,(352,352))

    def test_03_reject_float64(self):
        path,digest=self.file('fp64',np.zeros((352,352),np.float64))
        with self.assertRaises(ValueError):o.read_probability(path,digest,(352,352))

    def test_04_reject_wrong_shape_before_array_load(self):
        path,digest=self.file('shape',np.zeros((32,32),np.float32))
        with self.assertRaises(ValueError):o.read_probability(path,digest,(352,352))

    def test_05_reject_trailing_bytes(self):
        path,digest=self.file('trailing',np.zeros((352,352),np.float32),b'x')
        with self.assertRaises(ValueError):o.read_probability(path,digest,(352,352))

    def test_06_reject_fortran(self):
        path,digest=self.file('fortran',np.asfortranarray(np.zeros((352,352),np.float32)))
        with self.assertRaises(ValueError):o.read_probability(path,digest,(352,352))

    def test_07_reject_png_encoding(self):
        path,digest=self.file('uint8',np.zeros((352,352),np.uint8))
        with self.assertRaises(ValueError):o.read_probability(path,digest,(352,352))

    def test_08_reject_subset_before_pixels(self):
        with self.assertRaises(ValueError):o.validate_items([],['a'])

    def test_09_reject_relative_escape(self):
        with self.assertRaises(ValueError):o.validate_items([{'sample_id':'a','path':'../a.npy','sha256':'0'*64}],['a'])

    def test_10_reject_absent_digest(self):
        with self.assertRaises(ValueError):o.validate_items([{'sample_id':'a','path':'probabilities/a.npy'}],['a'])

    def test_11_preserve_member_order(self):
        items=[{'sample_id':x,'path':'probabilities/'+x+'.npy','sha256':'0'*64} for x in ('b','a')]
        with self.assertRaises(ValueError):o.validate_items(items,['a','b'])

    def test_12_reject_nonfinite(self):
        x=np.zeros((352,352),np.float32);x[0,0]=np.nan
        path,digest=self.file('nonfinite',x)
        with self.assertRaises(ValueError):o.read_probability(path,digest,(352,352))
