"""Offline evaluator for admitted stage41 exports; no inference or training entry."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import evaluation as e

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path('<LOCAL_HOST_PATH_03>')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_sources(contract):
    for path,digest in contract['source_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Evaluation source/definition changed: '+path)


def validate_items(items, expected_ids):
    if [x.get('sample_id') for x in items] != list(expected_ids):
        raise ValueError('Full frozen test membership/order required before reading pixels')
    for item in items:
        if item.get('path') != 'probabilities/'+item['sample_id']+'.npy':
            raise ValueError('Canonical prediction path required')
        digest=item.get('sha256','')
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Prediction SHA256 required')


def read_probability(path, expected_sha, raw_grid):
    if Path(path).stat().st_size > 416*416*4+4096:
        raise ValueError('Prediction file exceeds bounded format')
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected_sha:raise ValueError('Prediction bytes changed')
    stream=io.BytesIO(payload)
    if np.lib.format.read_magic(stream)!=(1,0):raise ValueError('Only bounded NPY v1 admitted')
    shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream,max_header_size=4096)
    if tuple(raw_grid) not in ((352,352),(416,416)) or shape!=tuple(raw_grid) or fortran or dtype!=np.dtype('<f4'):
        raise ValueError('NPY shape/order/type must be exact raw FP32 grid')
    if len(payload)-stream.tell()!=np.prod(shape)*4:raise ValueError('Truncated or trailing prediction bytes')
    array=np.load(io.BytesIO(payload),allow_pickle=False,max_header_size=4096)
    if not np.isfinite(array).all() or np.any((array<0)|(array>1)):raise ValueError('Not finite probabilities')
    return array


def admit_stage41(bundle, bundle_path, contract):
    if bundle.get('evaluation_contract_sha256')!=sha(HERE/'evaluation_contract.json'):
        raise ValueError('Unbound evaluation contract')
    if bundle.get('encoding')!='raw_fp32_sigmoid_no_quantization_no_minmax':
        raise ValueError('Probability representation not explicitly declared')
    if bundle.get('source_stage')!=41 or bundle.get('raw_grid')!=[352,352]:
        raise ValueError('PGSNet source not formally admitted by this RGBP entry')
    sys.path.insert(0,str(ROOT/'41_rgbp_portable_runner'))
    import rgbp_protocol as protocol
    formal=protocol.validate(protocol.read(protocol.PROTOCOL))
    matching=[x for x in formal['configs'] if x['config']['run_id']==bundle.get('run_id')]
    if len(matching)!=1:raise ValueError('Unknown or development run')
    item=matching[0];cfg=item['config'];out=Path(cfg['output_dir'])
    expected=(out/'common_grid45_exports/bundle.json').resolve()
    if Path(bundle_path).resolve()!=expected:raise ValueError('Export must be under its admitted run')
    if not protocol.core.completed_run(cfg,item['identity'],sha(protocol.PROTOCOL)):
        raise ValueError('Full formal training budget not verified')
    if bundle.get('best_checkpoint_sha256')!=sha(out/'best.pt'):
        raise ValueError('Not the validation-selected checkpoint')
    if bundle.get('completion_sha256')!=sha(out/'portable_complete.json'):
        raise ValueError('Run completion changed')
    return cfg


def evaluate(bundle_path):
    contract=read(HERE/'evaluation_contract.json')
    verify_sources(contract)
    bundle_path=Path(bundle_path).resolve(strict=True)
    bundle=read(bundle_path)
    expected=[x['sample_id'] for x in contract['members']]
    validate_items(bundle.get('items',[]),expected)
    cfg=admit_stage41(bundle,bundle_path,contract)
    output=HERE/'evaluations'/cfg['run_id']
    if output.exists():raise FileExistsError('Exclusive output; preserve prior evaluations')
    output.mkdir(parents=True)
    rows=[]
    try:
        for member,item in zip(contract['members'],bundle['items']):
            path=(bundle_path.parent/item['path']).resolve(strict=True)
            if not path.is_relative_to(bundle_path.parent):raise ValueError('Prediction escaped run directory')
            probability=read_probability(path,item['sha256'],bundle['raw_grid'])
            mask_path=(DATA/member['mask']).resolve(strict=True)
            if not mask_path.is_relative_to(DATA.resolve()):raise ValueError('Mask escaped dataset root')
            payload=mask_path.read_bytes()
            if hashlib.sha256(payload).hexdigest()!=member['mask_sha256']:raise ValueError('Frozen mask changed')
            mask=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_UNCHANGED)
            if mask is None:raise ValueError('Mask decode failed')
            rows.append(e.image_metrics(item['sample_id'],probability,mask,raw_grid=bundle['raw_grid']))
        report=e.complete_report(rows,expected,{x['sample_id']:x['stratum'] for x in contract['members']})
        report.update(run_id=cfg['run_id'],method=cfg['method'],seed=cfg['seed'],
            protocol_variant=cfg['protocol_variant'],evaluation_contract_sha256=sha(HERE/'evaluation_contract.json'),
            prediction_bundle_sha256=sha(bundle_path),best_checkpoint_sha256=bundle['best_checkpoint_sha256'])
        verify_sources(contract)
        with (output/'per_image.json').open('x',encoding='utf-8') as f:json.dump(rows,f,indent=2,allow_nan=False)
        with (output/'summary.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    except Exception as ex:
        with (output/'failure.json').open('x',encoding='utf-8') as f:
            json.dump({'error':type(ex).__name__+': '+str(ex),'completed':False,'rows_computed':len(rows)},f)
        raise
    with (output/'complete.json').open('x',encoding='utf-8') as f:
        json.dump({'evaluation_contract_sha256':sha(HERE/'evaluation_contract.json'),
                   'per_image_sha256':sha(output/'per_image.json'),'summary_sha256':sha(output/'summary.json')},f,indent=2)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',required=True)
    args=parser.parse_args();print(json.dumps(evaluate(args.bundle),indent=2))
