"""Bounded protocol2 structure reader; GLOBAL/REDUCE never invoke Python code."""
from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'54_pcod_safe_checkpoint_adapter'))
import metadata54 as m
need = m.need
MARK = object()


@dataclass(frozen=True)
class Symbol:
    name: str


@dataclass(frozen=True)
class Storage:
    key: str
    dtype: str
    count: int


@dataclass(frozen=True)
class Tensor:
    storage: Storage
    shape: tuple
    stride: tuple


@dataclass
class DType:
    built: bool = False


@dataclass
class Array:
    shape: tuple = ()
    data: bytes = b''
    built: bool = False


STORAGES = {'torch FloatStorage':'<f4', 'torch LongStorage':'<i8', 'torch ByteStorage':'|u1'}
GLOBALS = set(STORAGES) | {'torch.torch_version TorchVersion', 'torch._utils _rebuild_tensor_v2',
    'collections OrderedDict', 'numpy._core.multiarray _reconstruct', 'numpy ndarray', 'numpy dtype', '_codecs encode'}


def invoke(symbol, args):
    need(type(symbol) is Symbol and type(args) is tuple, 'Recognized symbolic calls only')
    name = symbol.name
    if name == 'collections OrderedDict':
        need(args == (), 'Empty ordered mapping only'); return OrderedDict()
    if name == '_codecs encode':
        need(len(args) == 2 and type(args[0]) is str and args[1] == 'latin1', 'Literal latin1 bytes only')
        return args[0].encode('latin1')
    if name == 'numpy dtype':
        need(args == ('u4', False, True), 'Only observed RNG uint32 dtype'); return DType()
    if name == 'numpy._core.multiarray _reconstruct':
        need(args == (Symbol('numpy ndarray'), (0,), b'b'), 'Observed NumPy shell only'); return Array()
    if name == 'torch._utils _rebuild_tensor_v2':
        need(len(args) == 6 and type(args[0]) is Storage and type(args[1]) is int and args[1] == 0, 'Full zero-offset storage only')
        storage, _, shape, stride, gradient, hooks = args
        need(type(shape) is tuple and type(stride) is tuple and len(shape) == len(stride) <= 8
             and all(type(x) is int and x >= 0 for x in shape) and type(gradient) is bool
             and isinstance(hooks, dict) and not hooks, 'Bounded dense recipe')
        expected = []; size = 1
        for dim in reversed(shape): expected.append(size); size *= dim
        need(stride == tuple(reversed(expected)) and size == storage.count, 'Contiguous exact storage coverage only')
        return Tensor(storage, shape, stride)
    raise ValueError('Unsupported symbolic reduction: ' + name)


def parse(payload):
    ops = m.operations(payload)
    stack, memo, globals_seen = [], {}, set()
    versions = 0

    def marked():
        indices = [i for i, value in enumerate(stack) if value is MARK]
        need(bool(indices), 'Missing MARK')
        i = indices[-1]; values = stack[i+1:]; del stack[i:]; return values

    def pairs(target, values):
        need(isinstance(target, dict) and len(values) % 2 == 0, 'Mapping pair count')
        for key, value in zip(values[::2], values[1::2]):
            need(type(key) in (str, int) and key not in target, 'Unique plain mapping keys')
            target[key] = value

    for op, arg, position in ops:
        name = op.name
        need(len(stack) <= 100_000 and len(memo) <= 100_000, 'Bounded symbolic state')
        if name == 'PROTO': continue
        if name == 'MARK': stack.append(MARK)
        elif name == 'NONE': stack.append(None)
        elif name in ('NEWTRUE','NEWFALSE'): stack.append(name == 'NEWTRUE')
        elif name in ('BININT','BININT1','BININT2','LONG1','LONG4','BINFLOAT','BINUNICODE'):
            need(type(arg) in (int,float,str), 'Plain literal')
            if type(arg) is float: need(math.isfinite(arg), 'Nonfinite metadata literal')
            stack.append(arg)
        elif name == 'EMPTY_DICT': stack.append({})
        elif name == 'EMPTY_LIST': stack.append([])
        elif name == 'EMPTY_TUPLE': stack.append(())
        elif name in ('BINPUT','LONG_BINPUT'):
            need(stack and type(arg) is int and arg not in memo, 'Unique initialized memo'); memo[arg] = stack[-1]
        elif name in ('BINGET','LONG_BINGET'):
            need(arg in memo, 'Missing memo'); stack.append(memo[arg])
        elif name == 'GLOBAL':
            need(arg in GLOBALS, 'Unsupported global token: ' + arg)
            globals_seen.add(arg); stack.append(Symbol(arg))
        elif name in ('TUPLE1','TUPLE2','TUPLE3'):
            count = int(name[-1]); need(len(stack) >= count, 'Tuple underflow')
            values = tuple(stack[-count:]); del stack[-count:]; stack.append(values)
        elif name == 'TUPLE': stack.append(tuple(marked()))
        elif name == 'APPENDS':
            values = marked(); need(stack and type(stack[-1]) is list, 'List target'); stack[-1].extend(values)
        elif name == 'APPEND':
            value = stack.pop(); need(stack and type(stack[-1]) is list, 'Append target'); stack[-1].append(value)
        elif name == 'SETITEMS':
            values = marked(); need(bool(stack), 'Mapping target'); pairs(stack[-1], values)
        elif name == 'SETITEM':
            value, key = stack.pop(), stack.pop(); need(bool(stack), 'Mapping target'); pairs(stack[-1], [key,value])
        elif name == 'BINPERSID':
            value = stack.pop()
            need(type(value) is tuple and len(value) == 5 and value[0] == 'storage'
                 and type(value[1]) is Symbol and value[1].name in STORAGES and type(value[2]) is str
                 and value[2].isdigit() and value[3] == 'cpu' and type(value[4]) is int and value[4] > 0,
                 'Explicit native CPU storage token only')
            stack.append(Storage(value[2], STORAGES[value[1].name], value[4]))
        elif name == 'REDUCE':
            args, symbol = stack.pop(), stack.pop(); stack.append(invoke(symbol,args))
        elif name == 'NEWOBJ':
            args, symbol = stack.pop(), stack.pop()
            need(symbol == Symbol('torch.torch_version TorchVersion') and type(args) is tuple
                 and len(args) == 1 and type(args[0]) is str, 'Version subclass token only')
            versions += 1; need(versions == 1, 'Single version constructor'); stack.append(args[0])
        elif name == 'BUILD':
            state = stack.pop(); need(bool(stack), 'Build target'); target = stack[-1]
            if type(target) is DType:
                need(not target.built and state == (3,'<',None,None,None,-1,-1,0), 'Exact uint32 dtype state')
                target.built = True
            elif type(target) is Array:
                need(not target.built and type(state) is tuple and len(state) == 5 and state[0] == 1
                     and type(state[1]) is tuple and len(state[1]) == 1 and type(state[1][0]) is int
                     and 0 < state[1][0] <= 10000 and type(state[2]) is DType and state[2].built
                     and state[3] is False and type(state[4]) is bytes and len(state[4]) == 4 * state[1][0],
                     'Exact bounded uint32 vector state')
                target.shape, target.data, target.built = state[1], state[4], True
            elif type(target) is OrderedDict:
                need(type(state) is dict and set(state) == {'_metadata'} and isinstance(state['_metadata'],dict),
                     'Only optional ordered-state metadata')
                m.store.json_bytes(state['_metadata'])
                target._metadata = state['_metadata']
            else: raise ValueError('Unsupported symbolic BUILD target')
        elif name == 'STOP':
            need(len(stack) == 1 and isinstance(stack[0],dict) and versions in (0,1), 'Single complete root/version')
            return stack[0], {'instructions':len(ops), 'globals':sorted(globals_seen), 'executed_global_callables':0}
        else: raise ValueError('Unsupported opcode: ' + name)
    raise ValueError('Missing terminal root')


def tensors(value, depth=0, ancestors=None):
    need(depth < 30, 'Bounded graph depth')
    ancestors = set() if ancestors is None else ancestors
    if type(value) is Tensor: yield value
    elif isinstance(value,(dict,list,tuple)):
        need(id(value) not in ancestors, 'Cyclic graph')
        ancestors = ancestors | {id(value)}
        for x in (value.values() if isinstance(value,dict) else value): yield from tensors(x,depth+1,ancestors)
    else: need(type(value) in (str,int,float,bool,type(None),Array), 'Leftover symbolic object')


class Archive:
    def __init__(self,path):
        self.zip = zipfile.ZipFile(path)
        infos, metadata = m.archive_layout(self.zip)
        self.prefix = metadata.rsplit('/',1)[0]
        self.root, self.parse_report = parse(self.zip.read(metadata))
        self.storage = {}
        for tensor in tensors(self.root):
            storage = tensor.storage
            need(storage.key not in self.storage or self.storage[storage.key] == storage, 'Inconsistent storage alias')
            self.storage[storage.key] = storage
        expected = {self.prefix+'/data/'+key for key in self.storage}
        actual = {i.filename for i in infos if i.filename.startswith(self.prefix+'/data/')}
        need(actual == expected, 'All and only referenced storage members')
        for key, storage in self.storage.items():
            need(self.zip.getinfo(self.prefix+'/data/'+key).file_size == storage.count*np.dtype(storage.dtype).itemsize, 'Storage size differs')

    def close(self): self.zip.close()

    def tensor_hash(self,tensor):
        need(type(tensor) is Tensor, 'Tensor token required')
        with self.zip.open(self.prefix+'/data/'+tensor.storage.key) as stream: return m.stream_sha(stream)

    def scalar(self,tensor):
        need(type(tensor) is Tensor and tensor.shape == (), 'Scalar tensor required')
        with self.zip.open(self.prefix+'/data/'+tensor.storage.key) as stream:
            return np.frombuffer(stream.read(),dtype=tensor.storage.dtype).item()

    def finite_storages(self):
        evidence = {}
        for key, storage in self.storage.items():
            digest = m.hashlib.sha256(); size = 0
            with self.zip.open(self.prefix+'/data/'+key) as stream:
                for block in iter(lambda:stream.read(1024**2),b''):
                    need(np.isfinite(np.frombuffer(block,dtype=storage.dtype)).all(), 'Nonfinite storage')
                    size += len(block); digest.update(block)
            evidence[key] = {'dtype':storage.dtype,'count':storage.count,'bytes':size,'sha256':digest.hexdigest()}
        return evidence
