"""PCOD input/metric/statistic primitives. Importing does not read dataset pixels."""
from fractions import Fraction
import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEEDS = (42, 2026, 3407, 7819, 12011)
METHODS = ('direct_reference', 'pfe_reference', 'learned_fusion_reference',
           'global_film_style', 'wide_direct', 'deeplabv3plus_resnet50')
GRID = [352, 352]
ENCODING = 'model_device_sigmoid_then_cpu_fp32_no_quantization_no_minmax'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


metrics = load('pcod53_shared_metrics45', ROOT / '45_rgbp_common_evaluation/evaluation.py')
sys.path.insert(0, str(ROOT / '45_rgbp_common_evaluation'))
npy = load('pcod53_npy45', ROOT / '45_rgbp_common_evaluation/offline_evaluate.py')
sha, read_probability = npy.sha, npy.read_probability


def read(path):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError('Duplicate JSON field')
            result[k] = v
        return result
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      object_pairs_hook=pairs, parse_constant=reject)


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def validate_ids(ids):
    if (not ids or len(set(ids)) != len(ids) or
            any(not isinstance(x, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,180}', x) for x in ids)):
        raise ValueError('Nonempty unique canonical PCOD stems required')


def bound_payload(root, member, modality):
    root = Path(root).resolve(strict=True)
    relative = Path(member[modality])
    path = (root / relative).resolve(strict=True)
    if relative.is_absolute() or not path.is_relative_to(root) or path.stat().st_size > 64 * 1024**2:
        raise ValueError('Image path/size boundary')
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != member[modality + '_sha256']:
        raise ValueError('Frozen image bytes differ')
    return payload


def pil_grid(payload, mode, interpolation):
    with Image.open(io.BytesIO(payload)) as image:
        if image.width * image.height > 16_000_000 or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Image pixel/frame boundary')
        return np.asarray(image.convert(mode).resize((352, 352), interpolation), dtype=np.float32)


def input_tensor(rgb_bytes, dop_bytes):
    rgb = pil_grid(rgb_bytes, 'RGB', Image.Resampling.BILINEAR) / 255.0
    dop = pil_grid(dop_bytes, 'L', Image.Resampling.BILINEAR) / 255.0
    return torch.from_numpy(np.concatenate((rgb.transpose(2, 0, 1), dop[None]), axis=0).astype(np.float32))


def mask_grid(payload):
    # PCOD resizes first, unlike RGBP's binarize-then-OpenCV convention.
    return (pil_grid(payload, 'L', Image.Resampling.NEAREST) > 127).astype(np.float32)


class BoundInputs(torch.utils.data.Dataset):
    def __init__(self, root, members):
        validate_ids([m['sample_id'] for m in members])
        self.root, self.members = root, members

    def __len__(self):
        return len(self.members)

    def __getitem__(self, index):
        member = self.members[index]
        return {'image': input_tensor(bound_payload(self.root, member, 'rgb'),
                                      bound_payload(self.root, member, 'dop')),
                'stem': member['sample_id']}


def probabilities(logits, count):
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (count, 1, 352, 352):
        raise ValueError('Exact one-head 352-grid logits required')
    if not torch.isfinite(logits.detach().float().cpu()).all():
        raise ValueError('Nonfinite logits must not be hidden by sigmoid saturation')
    return torch.sigmoid(logits.detach()).float().cpu().numpy()[:, 0]


def image_metrics(stem, probability, mask):
    validate_ids([stem])
    if (probability.dtype != np.float32 or probability.shape != (352, 352)
            or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1))):
        raise ValueError('Raw finite FP32 PCOD probabilities required')
    if mask.dtype != np.float32 or mask.shape != (352, 352) or not np.isin(mask, (0, 1)).all():
        raise ValueError('Already transformed binary FP32 PCOD mask required')
    row = metrics.boundary.pair_metrics(probability > .5, mask > .5,
                                        tolerances=(1, 2, 4), band_ratio=.02)
    row.update(sample_id=stem, mae=float((torch.from_numpy(probability) - torch.from_numpy(mask)).abs().mean()),
               evaluated_pixels=352 * 352)
    return row


aggregate = metrics.aggregate


def seed_summary(values, seeds):
    if (set(values) != set(seeds) or any(type(s) is not int for s in values)
            or any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1 for v in values.values())):
        raise ValueError('Exact seed IDs and finite unit-interval metrics required')
    a = [float(values[s]) for s in seeds]
    return {'seeds': list(seeds), 'values': a, 'mean': float(np.mean(a)),
            'sample_sd': float(np.std(a, ddof=1)), 'minimum': min(a), 'maximum': max(a), 'n': len(a)}


def paired(pfe, comparator, method, *, sign_flip=False):
    if method not in METHODS or method == 'pfe_reference':
        raise ValueError('Registered comparator required')
    seeds = SEEDS[:3] if method == METHODS[-1] else SEEDS
    a, b = seed_summary(pfe, seeds), seed_summary(comparator, seeds)
    d = [x - y for x, y in zip(a['values'], b['values'])]
    result = {'pfe': a, 'comparator': b, 'pfe_minus_comparator': d,
              'mean': float(np.mean(d)), 'sample_sd': float(np.std(d, ddof=1)),
              'minimum': min(d), 'maximum': max(d), 'n': len(seeds),
              'replication_unit': 'training_run', 'same_batch_order_implied': False}
    if sign_flip:
        exact = [Fraction.from_float(x) for x in d]
        observed = abs(sum(exact))
        count = sum(abs(sum(s * x for s, x in zip(signs, exact))) >= observed
                    for signs in itertools.product((-1, 1), repeat=len(seeds)))
        result.update(two_sided_non_mid_p=count / 2**len(seeds), enumerations=2**len(seeds),
                      assumption='joint sign exchangeability, not guaranteed by matched seed labels')
    return result


def statistics(reports):
    expected = {(m, s) for m in METHODS for s in (SEEDS[:3] if m == METHODS[-1] else SEEDS)}
    index = {(r['method'], r['seed']): r['all_test'] for r in reports}
    if len(reports) != 28 or set(index) != expected:
        raise ValueError('Exactly all 28 completed formal evaluations required')
    keys = ('iou', 'f1', 'image_mae', 'boundary_f1_2px_image_mean', 'boundary_iou_pooled')
    contrasts = {}
    for m in METHODS:
        if m == 'pfe_reference':
            continue
        seeds = SEEDS[:3] if m == METHODS[-1] else SEEDS
        contrasts[m] = {k: paired({s: index['pfe_reference', s][k] for s in seeds},
                                  {s: index[m, s][k] for s in seeds}, m,
                                  sign_flip=(m == 'direct_reference' and k == 'iou')) for k in keys}
    return {'contrasts': contrasts, 'primary': 'PFE minus Direct pooled IoU, five matched seeds',
            'secondary': 'descriptive, no unadjusted confirmatory superiority claim',
            'mae_direction': 'negative PFE-minus-comparator favors PFE',
            'physical_scene_independence_established': False,
            'images_or_boundary_pixels_are_training_replicates': False}


def write_probability(path, value):
    if value.dtype != np.float32 or value.shape != (352, 352) or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError('Finite raw 352 FP32 probability required')
    with Path(path).open('xb') as stream:
        np.lib.format.write_array(stream, np.ascontiguousarray(value, dtype='<f4'), version=(1, 0), allow_pickle=False)
    digest = sha(path)
    if not np.array_equal(read_probability(path, digest, GRID), value):
        raise ValueError('Probability serialization changed values')
    return digest


def export_batches(model, loader, device, output, ids, provenance, final_check):
    validate_ids(ids)
    if {'items', 'encoding', 'raw_grid'} & provenance.keys():
        raise ValueError('Reserved provenance fields')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'probabilities').mkdir()
    items = []
    try:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                if set(batch) != {'image', 'stem'}:
                    raise ValueError('Mask-free inference batches required')
                stems = list(batch['stem'])
                if not stems or stems != ids[len(items):len(items) + len(stems)]:
                    raise ValueError('Missing, extra, duplicated or reordered members')
                image = batch['image']
                if (image.dtype != torch.float32 or tuple(image.shape) != (len(stems), 4, 352, 352)
                        or not torch.isfinite(image).all()):
                    raise ValueError('PCOD four-channel finite input required')
                values = probabilities(model(image.to(device)), len(stems))
                for stem, value in zip(stems, values):
                    relative = 'probabilities/' + stem + '.npy'
                    items.append({'sample_id': stem, 'path': relative,
                                  'sha256': write_probability(output / relative, value)})
        if len(items) != len(ids):
            raise ValueError('Incomplete inference population')
        final_check()
        bundle = dict(provenance, items=items, raw_grid=GRID, encoding=ENCODING)
        save_new(output / 'bundle.json', bundle)
        save_new(output / 'export_complete.json', {'bundle_sha256': sha(output / 'bundle.json'),
                 'contract_sha256': provenance['contract_sha256'], 'images': len(ids), 'stage': 53})
        return bundle
    except BaseException as error:
        save_new(output / 'failure.json', {'error': type(error).__name__ + ': ' + str(error),
                                          'images_written': len(items), 'automatic_retry': False})
        raise


def verify_receipt(bundle_path, contract_sha, ids):
    validate_ids(ids)
    path = Path(bundle_path).resolve(strict=True)
    if path.name != 'bundle.json' or (path.parent / 'failure.json').exists():
        raise ValueError('Failed or noncanonical export')
    bundle, receipt = read(path), read(path.parent / 'export_complete.json')
    if (receipt != {'bundle_sha256': sha(path), 'contract_sha256': contract_sha, 'images': len(ids), 'stage': 53}
            or bundle.get('contract_sha256') != contract_sha or bundle.get('encoding') != ENCODING
            or bundle.get('raw_grid') != GRID):
        raise ValueError('Unbound export receipt/representation')
    npy.validate_items(bundle.get('items', []), ids)
    expected = {'bundle.json', 'export_complete.json'} | {x['path'] for x in bundle['items']}
    if {p.relative_to(path.parent).as_posix() for p in path.parent.rglob('*') if p.is_file()} != expected:
        raise ValueError('Export has unexpected/missing files')
    for item in bundle['items']:
        prediction = (path.parent / item['path']).resolve(strict=True)
        if not prediction.is_relative_to(path.parent):
            raise ValueError('Prediction path escape')
        read_probability(prediction, item['sha256'], GRID)
    return bundle
