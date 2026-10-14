"""Mask-free inputs and exclusive raw-probability transactions; no device entry."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / '45_rgbp_common_evaluation'))
import offline_evaluate as offline


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


read, sha = offline.read, offline.sha


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def validate_ids(ids):
    if not ids or len(ids) != len(set(ids)) or any(not re.fullmatch(r'[0-9]{14}', s) for s in ids):
        raise ValueError('Unique canonical nonempty sample IDs required')


def input_tensor(rgb, dolp, mode, size):
    if rgb.dtype != np.uint16 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected native three-channel uint16 RGB')
    if dolp.dtype != np.uint8 or dolp.shape != rgb.shape:
        raise ValueError('Expected matched three-channel uint8 DoLP')
    if mode not in ('rgb', 'rgb_dolp') or type(size) is not int or size < 1:
        raise ValueError('Unsupported input contract')
    rgb = rgb[..., ::-1].astype(np.float32) / np.float32(65535)
    dolp = dolp.astype(np.float32) / np.float32(255)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    dolp = cv2.resize(dolp, (size, size), interpolation=cv2.INTER_LINEAR)
    value = rgb if mode == 'rgb' else np.concatenate((rgb, dolp), axis=2)
    return torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1)))


class BoundInputs(torch.utils.data.Dataset):
    """Hashes the exact compressed bytes decoded; no masks are opened or returned."""
    def __init__(self, root, members, mode, size):
        self.root = Path(root).resolve(strict=True)
        self.members, self.mode, self.size = members, mode, size
        validate_ids([m['sample_id'] for m in members])

    def __len__(self):
        return len(self.members)

    def __getitem__(self, index):
        member = self.members[index]
        arrays = []
        for name, suffix in (('image', 'rgb'), ('dolp', 'dolp')):
            expected = f"test/{name}/{member['sample_id']}_{suffix}.tiff"
            if member[name] != expected:
                raise ValueError('Noncanonical input path')
            path = (self.root / expected).resolve(strict=True)
            if not path.is_relative_to(self.root) or path.stat().st_size > 64 * 1024**2:
                raise ValueError('Input path or compressed-size boundary')
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != member[name + '_sha256']:
                raise ValueError('Frozen input bytes changed')
            array = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
            if array is None or array.shape[0] * array.shape[1] > 16_000_000:
                raise ValueError('Input decode or pixel boundary failed')
            arrays.append(array)
        return {'image': input_tensor(*arrays, self.mode, self.size), 'stem': member['sample_id']}


def probabilities(logits, count, grid):
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (count, 1, *grid):
        raise ValueError('Expected one dense logit head on the exact raw grid')
    value = logits.detach().float().cpu()
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite logits must not be hidden by sigmoid saturation')
    return torch.sigmoid(value).numpy()[:, 0]


def write_probability(path, array):
    if array.dtype != np.float32 or array.ndim != 2 or not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError('Finite raw FP32 probabilities required')
    with Path(path).open('xb') as stream:
        np.lib.format.write_array(stream, np.ascontiguousarray(array, dtype='<f4'), version=(1, 0), allow_pickle=False)
    return sha(path)


def export_batches(model, loader, device, output, ids, grid, provenance, final_check):
    validate_ids(ids)
    if grid not in ([352, 352], [416, 416]):
        raise ValueError('Only bounded public raw grids supported')
    reserved = {'items', 'raw_grid', 'encoding'}
    if reserved & provenance.keys():
        raise ValueError('Caller cannot override computed export fields')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'probabilities').mkdir()
    items = []
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                if set(batch) != {'image', 'stem'}:
                    raise ValueError('Inference loader must be mask-free')
                stems = list(batch['stem'])
                if not stems or stems != ids[len(items):len(items) + len(stems)]:
                    raise ValueError('Missing, reordered, duplicate or extra inference members')
                image = batch['image']
                if image.dtype != torch.float32 or image.shape[0] != len(stems) or not torch.isfinite(image).all():
                    raise ValueError('Invalid inference tensor')
                outputs = probabilities(model(image.to(device)), len(stems), grid)
                for sid, value in zip(stems, outputs):
                    relative = 'probabilities/' + sid + '.npy'
                    digest = write_probability(output / relative, value)
                    # Verify serialized bytes, not only the in-memory array.
                    loaded = offline.read_probability(output / relative, digest, grid)
                    if not np.array_equal(loaded, value):
                        raise ValueError('Raw probability roundtrip changed values')
                    items.append({'sample_id': sid, 'path': relative, 'sha256': digest})
        if len(items) != len(ids):
            raise ValueError('Incomplete inference population')
        final_check()
        bundle = dict(provenance, items=items, raw_grid=grid,
                      encoding='raw_fp32_sigmoid_no_quantization_no_minmax')
        offline.validate_items(items, ids)
        save_new(output / 'bundle.json', bundle)
        save_new(output / 'export_complete.json', {
            'bundle_sha256': sha(output / 'bundle.json'), 'images': len(items),
            'export_contract_sha256': provenance['export_contract_sha256'],
            'test_metrics_computed': False, 'exporter_stage': 46})
        return bundle
    except BaseException as error:
        save_new(output / 'failure.json', {'error': type(error).__name__ + ': ' + str(error),
                                         'images_written': len(items), 'automatic_retry': False})
        raise


def verify_receipt(bundle_path, export_contract_sha, ids):
    bundle_path = Path(bundle_path).resolve(strict=True)
    if (bundle_path.parent / 'failure.json').exists():
        raise ValueError('Failed export cannot be admitted')
    bundle = read(bundle_path)
    receipt = read(bundle_path.parent / 'export_complete.json')
    if (bundle.get('export_contract_sha256') != export_contract_sha
            or receipt != {'bundle_sha256': sha(bundle_path), 'images': len(ids),
                           'export_contract_sha256': export_contract_sha,
                           'test_metrics_computed': False, 'exporter_stage': 46}):
        raise ValueError('Export transaction receipt differs')
    offline.validate_items(bundle.get('items', []), ids)
    for item in bundle['items']:
        path = (bundle_path.parent / item['path']).resolve(strict=True)
        if not path.is_relative_to(bundle_path.parent):
            raise ValueError('Prediction escaped export root')
        offline.read_probability(path, item['sha256'], bundle['raw_grid'])
    return bundle
