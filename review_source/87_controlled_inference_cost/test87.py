from pathlib import Path
import json,hashlib
from timing87 import timed_forward
H=Path(__file__).resolve().parent

def tests():
 events=[];tick=[100]
 def clock():events.append('clock');return tick[0]
 class Value:
  def sigmoid(self):events.append('sigmoid');tick[0]+=20;return self
  def cpu(self):events.append('blocking_readback');tick[0]+=700;return self
  def contiguous(self):events.append('contiguous');tick[0]+=5;return self
 def model(x):events.append('enqueue');tick[0]+=10;return Value()
 cpu,interval=timed_forward(model,None,clock)
 assert interval=={'start_ns':100,'end_ns':835,'elapsed_ns':735}
 assert events==['clock','enqueue','sigmoid','blocking_readback','contiguous','clock']
 # A delayed readback is included, while downstream CPU validation is outside timing.
 tick[0]+=99999;assert interval['elapsed_ns']==735
 bad=iter([100,99])
 try:timed_forward(model,None,lambda:next(bad))
 except ValueError as e:assert str(e)=='invalid_clock_interval'
 else:raise AssertionError('backwards clock not rejected')
 def broken(x):raise RuntimeError('deliberate_forward_error')
 try:timed_forward(broken,None,lambda:100)
 except RuntimeError as e:assert str(e)=='deliberate_forward_error'
 else:raise AssertionError('forward failure swallowed')
 return {'passed':True,'checks':['readback_and_contiguity_inside_timer','downstream_validation_outside_timer','backwards_clock_rejected','forward_failure_propagates'],'test_scope':'stdlib timing boundary tests, not DirectML synchronization or model correctness'}
if __name__=='__main__':
 result=tests();result['timing_source_sha256']=hashlib.sha256((H/'timing87.py').read_bytes()).hexdigest()
 with (H/'cpu_tests.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
 print(result)
