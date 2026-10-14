"""Completion-bounded forward timing; no torch import or automatic synchronization claims."""
import time

def timed_forward(model,device_input,clock=time.perf_counter_ns):
 start=clock()
 logits=model(device_input)
 probability=logits.sigmoid()
 cpu=probability.cpu().contiguous()
 end=clock()
 if not isinstance(start,int) or not isinstance(end,int) or end<=start:raise ValueError('invalid_clock_interval')
 del logits,probability
 return cpu,{'start_ns':start,'end_ns':end,'elapsed_ns':end-start}
