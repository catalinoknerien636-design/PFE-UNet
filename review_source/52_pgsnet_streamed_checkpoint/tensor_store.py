"""Bounded, pickle-free per-tensor checkpoint storage. No models or GPU launcher."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import numpy as np

HERE = Path(__file__).resolve().parent
FORMAT = 'pfe-pgs-technical-npy-v1'
MAX_TENSOR = 64 * 1024**2
MAX_TOTAL = 8 * 1024**3
MAX_META = 8 * 1024**2
DTYPES = ('<f4','<i8','|u1')


def need(value, message):
    if not value: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f: return stream_sha(f)


def stream_sha(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def object_sha(value): return hashlib.sha256(json_bytes(value)).hexdigest()


def no_duplicate(pairs):
    out={}
    for key,value in pairs:
        need(key not in out,'Duplicate JSON field')
        out[key]=value
    return out


def read_json(path):
    need(Path(path).stat().st_size <= MAX_META,'Metadata size cap')
    return json.loads(Path(path).read_bytes(),object_pairs_hook=no_duplicate,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def write_json(path,value):
    data=json_bytes(value); need(len(data) <= MAX_META,'Metadata size cap')
    with Path(path).open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())


def boundary(path, root):
    root, path=Path(root).resolve(),Path(path).resolve()
    need(root.is_relative_to(HERE) and root != HERE and path.parent == root,'Immediate new-stage output boundary')
    return path


def schema_row(key, shape, dtype):
    need(type(key) is str and 0 < len(key) <= 240 and re.fullmatch(r'[A-Za-z0-9_.]+',key),'Canonical tensor key')
    need(type(shape) is list and len(shape) <= 5 and all(type(n) is int and 0 < n <= 2**26 for n in shape),'Tensor dimensions')
    need(type(dtype) is str and dtype in DTYPES,'Only FP32, int64 and uint8; no pickle/object/complex dtype')
    size=math.prod(shape)*np.dtype(dtype).itemsize
    need(size <= MAX_TENSOR,'Per-tensor byte cap')
    return {'key':key,'shape':shape,'dtype':dtype,'bytes':size}


def validate_schema(schema):
    need(type(schema) is list and 0 < len(schema) <= 10000,'Bounded tensor count')
    keys=[]; total=0
    for row in schema:
        need(type(row) is dict and set(row) == {'key','shape','dtype','bytes'},'Tensor schema fields')
        expected=schema_row(row['key'],row['shape'],row['dtype'])
        need(object_sha(row) == object_sha(expected),'Typed tensor schema')
        keys.append(row['key']); total+=row['bytes']
    need(len(keys) == len(set(keys)),'Duplicate tensor key')
    need(total <= MAX_TOTAL,'Total tensor byte cap')
    return total


def finite(array):
    if array.dtype.kind == 'f':
        flat=array.reshape(-1)
        for start in range(0,flat.size,262144):
            need(np.isfinite(flat[start:start+262144]).all(),'Nonfinite tensor')


def validate_array(array,row):
    need(type(array) is np.ndarray and array.flags.c_contiguous,'Plain contiguous ndarray')
    need(array.dtype.str == row['dtype'] and list(array.shape) == row['shape'],'Tensor schema differs')
    finite(array)


def read_array(path,row,digest):
    need(Path(path).stat().st_size <= row['bytes']+4096,'Serialized tensor cap')
    with Path(path).open('rb') as f:
        need(stream_sha(f) == digest,'Tensor file hash changed')
        f.seek(0)
        need(np.lib.format.read_magic(f) == (1,0),'Only explicit NPY v1 format')
        shape,fortran,dtype=np.lib.format.read_array_header_1_0(f,max_header_size=4096)
        need(not fortran and list(shape) == row['shape'] and dtype.str == row['dtype'],'NPY header schema')
        need(f.tell()+row['bytes'] == Path(path).stat().st_size,'Exact tensor payload length; no trailing bytes')
        f.seek(0)
        value=np.lib.format.read_array(f,allow_pickle=False,max_header_size=4096)
        validate_array(value,row)
        f.seek(0); need(stream_sha(f) == digest,'Tensor file changed during read')
    return value


def write_bundle(path, root, schema, tensors, metadata, identity, after=lambda phase:None):
    total=validate_schema(schema)
    need(type(metadata) is dict and type(identity) is dict,'Explicit metadata and identity')
    need(len(json_bytes(metadata))+len(json_bytes(identity)) <= MAX_META//2,'Bounded plain JSON metadata')
    path=boundary(path,root)
    need(shutil.disk_usage(root).free >= 20*1024**3+total,'Keep20GiB disk reserve plus payload')
    path.mkdir(exist_ok=False)
    try:
        after('created'); items=iter(tensors); files=[]
        for i,row in enumerate(schema):
            key,value=next(items)
            need(key == row['key'],'Stream tensor ordering differs')
            validate_array(value,row)
            partial=path/f'{i:05d}.part'; final=path/f'{i:05d}.npy'
            with partial.open('xb') as f:
                np.lib.format.write_array(f,value,version=(1,0),allow_pickle=False)
                f.flush(); os.fsync(f.fileno())
            digest=sha(partial)
            # Read back one tensor, never collect the complete payload in RAM.
            reread=read_array(partial,row,digest)
            need(np.array_equal(value,reread),'Tensor write/read differs')
            del reread, value
            os.replace(partial,final)
            files.append({'file':final.name,'sha256':digest})
            after('tensor_'+str(i))
        try: next(items)
        except StopIteration: pass
        else: raise ValueError('Unexpected extra tensor')
        manifest={'format':FORMAT,'schema':schema,'files':files,'metadata':metadata,'identity':identity,'total_bytes':total}
        write_json(path/'manifest.json',manifest); after('manifest')
        receipt={'format':FORMAT,'manifest_sha256':sha(path/'manifest.json'),'identity_sha256':object_sha(identity)}
        write_json(path/'complete.part',receipt); after('receipt_flushed')
        os.replace(path/'complete.part',path/'complete.json')
        return sha(path/'complete.json')
    except Exception as error:
        write_json(path/'failure.json',{'error':type(error).__name__,'message':str(error)})
        raise


def preflight_bundle(path,root,receipt_sha,identity):
    path=boundary(path,root)
    need(type(receipt_sha) is str and re.fullmatch(r'[0-9a-f]{64}',receipt_sha),'Externally bound completion hash')
    receipt_path=path/'complete.json'
    need(receipt_path.is_file() and sha(receipt_path) == receipt_sha,'Missing or changed completion receipt')
    receipt=read_json(receipt_path)
    need(set(receipt) == {'format','manifest_sha256','identity_sha256'} and receipt['format'] == FORMAT,'Receipt format')
    need(receipt['identity_sha256'] == object_sha(identity),'Receipt identity changed')
    manifest_path=path/'manifest.json'
    need(sha(manifest_path) == receipt['manifest_sha256'],'Manifest changed')
    manifest=read_json(manifest_path)
    need(type(manifest) is dict and set(manifest) == {'format','schema','files','metadata','identity','total_bytes'},'Manifest fields')
    need(manifest['format'] == FORMAT and object_sha(manifest['identity']) == object_sha(identity),'Manifest identity')
    need(type(manifest['total_bytes']) is int and validate_schema(manifest['schema']) == manifest['total_bytes'],'Total bytes')
    files=manifest['files']; need(type(files) is list and len(files) == len(manifest['schema']),'Tensor file count')
    expected={'complete.json','manifest.json'}
    for i,(row,entry) in enumerate(zip(manifest['schema'],files)):
        need(type(entry) is dict and set(entry) == {'file','sha256'},'Tensor file entry')
        need(entry['file'] == f'{i:05d}.npy','No linked/traversing/renamed tensor path')
        need(type(entry['sha256']) is str and re.fullmatch(r'[0-9a-f]{64}',entry['sha256']),'Tensor digest')
        expected.add(entry['file'])
    need({p.name for p in path.iterdir()} == expected,'Incomplete, failed or foreign files in bundle')
    need(all(p.is_file() and not p.is_symlink() and p.resolve().parent == path for p in path.iterdir()),'External linked artifact')
    for row,entry in zip(manifest['schema'],files):
        value=read_array(path/entry['file'],row,entry['sha256']); del value
    return manifest


def iter_verified(path, manifest):
    for row,entry in zip(manifest['schema'],manifest['files']):
        value=read_array(Path(path)/entry['file'],row,entry['sha256'])
        yield row['key'],value
        del value
