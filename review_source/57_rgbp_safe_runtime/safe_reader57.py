"""Owned, bounded dense checkpoint reader; no pickle execution or fallback."""
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
import hashlib,json
import numpy as np
import torch
import symbolic57 as s

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()

class Reader:
 def __init__(self,entries=None,fixture_root=None):
  self.entries={Path(x['config']['output_dir']).resolve():x for x in (entries or [])}
  self.fixture_root=Path(fixture_root).resolve() if fixture_root else None
  self.calls=[]
 def __call__(self,file,map_location=None,pickle_module=None,*,weights_only=None,mmap=None,**kwargs):
  if not isinstance(file,(str,Path)) or map_location not in (None,'cpu') or pickle_module is not None or mmap or kwargs:
   raise ValueError('Only owned path and CPU map supported')
  path=Path(file).resolve()
  if path.name not in ('latest.pt','best.pt'):raise ValueError('Only latest/best owned checkpoint')
  entry=self.entries.get(path.parent)
  if entry is None and (self.fixture_root is None or not path.is_relative_to(self.fixture_root)):
   raise ValueError('Unbound checkpoint path')
  cfg=read_json(path.parent/'config.json');identity=read_json(path.parent/'identity.json')
  if entry is not None and (cfg!=entry['config'] or identity!=entry['identity']):raise ValueError('Bound sidecars differ')
  if path.stat().st_size>s.m.MAX_ARCHIVE:raise ValueError('Archive size cap')
  before=digest(path);archive=s.Archive(path)
  try:
   if archive.root.get('identity')!=identity:raise ValueError('Parsed identity differs')
   archive.finite_storages()
   cache={}
   def convert(value):
    if type(value) is s.Tensor:
     key=value.storage.key
     if key not in cache:
      raw=archive.zip.read(archive.prefix+'/data/'+key)
      cache[key]=torch.from_numpy(np.frombuffer(raw,dtype=value.storage.dtype).copy())
     return cache[key].reshape(value.shape)
    if type(value) is s.Array:
     if not value.built:raise ValueError('Unbuilt NumPy literal')
     return np.frombuffer(value.data,dtype='<u4').copy().reshape(value.shape)
    if isinstance(value,dict):
     out=OrderedDict() if type(value) is OrderedDict else {}
     for key,item in value.items():out[key]=convert(item)
     if hasattr(value,'_metadata'):out._metadata=convert(value._metadata)
     return out
    if type(value) is list:return [convert(x) for x in value]
    if type(value) is tuple:return tuple(convert(x) for x in value)
    if type(value) in (str,int,float,bool,type(None)):return value
    raise ValueError('Unsupported parsed node')
   result=convert(archive.root)
  finally:archive.close()
  if digest(path)!=before:raise ValueError('Checkpoint changed during safe read')
  if read_json(path.parent/'config.json')!=cfg or read_json(path.parent/'identity.json')!=identity:raise ValueError('Sidecars changed during read')
  self.calls.append(dict(path=str(path),sha256=before,pickle_executed=False))
  return result

@contextmanager
def installed(reader):
 old_load,old_serialization=torch.load,torch.serialization.load
 torch.load=reader;torch.serialization.load=reader
 try:yield reader
 finally:torch.load=old_load;torch.serialization.load=old_serialization
