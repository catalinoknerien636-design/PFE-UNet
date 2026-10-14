"""Explicit technical-only pretraining contract; original sources remain unchanged."""
import ast
from collections import OrderedDict
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UPSTREAM = ROOT / '26_pgsnet_audit/upstream'
VENDOR = ROOT / '16_standard_backbone/vendor/timm/layers'
CPU_ENV = ROOT / '31_pgsnet_checkpoint_safety/venv_cpu'
RESNET = ROOT / '35_pgsnet_resnet_type_audit/tensor_candidate_v1/resnet18_imagenet1k_v1_TENSOR_ONLY_NOT_ADOPTED.safetensors'
CONFORMER = ROOT / '31_pgsnet_checkpoint_safety/external_backbones_proxy/Conformer_base_patch16.pth'
RESNET_SHA = '470403f13b8796191b680bef15a4d7ff34dd8384f292c6e1f21ee57bdbcb9b4a'
CONFORMER_SHA = '1f497f76958389238c83cb4d4f90a7b27acfb76a713075d328fe8b07f987e791'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)


def require_runtime():
    import torch
    import safetensors
    if (Path(sys.prefix).resolve() != CPU_ENV.resolve() or torch.__version__ != '2.14.0+cpu'
            or torch.version.cuda is not None or safetensors.__version__ != '0.6.2'
            or not Path(torch.__file__).resolve().is_relative_to(CPU_ENV.resolve())):
        raise RuntimeError('Use the exact independent reviewed CPU runtime')
    return torch


def check_files(rows):
    for name, expected in rows.items():
        if digest(name) != expected:
            raise RuntimeError('Frozen file changed: ' + name)


def validate_contract():
    plan = read(HERE / 'contract.json')
    if (plan['formal_adoption'] or plan['gpu_authorized'] or plan['dataset_training_authorized']
            or plan['resnet_counter_initialization'] != 'explicit_zero_int64_for_exact_20_missing_keys'
            or plan['equivalent_to_author_resnet'] != 'not_established'):
        raise ValueError('Technical-only policy changed')
    check_files(plan['protected_files'])
    check_files(plan['implementation_sha256'])
    return plan


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resnet_module():
    return module_from_file('pfe36_resnet', UPSTREAM / 'backbone/resnet/resnet.py')


def exact_state(state, expected, torch):
    if type(state) not in (dict, OrderedDict) or set(state) != set(expected):
        raise ValueError('Exact state keys required before loading')
    for name, value in state.items():
        if (type(value) is not torch.Tensor or value.device.type != 'cpu'
                or value.layout != torch.strided or value.is_quantized
                or value.shape != expected[name].shape or value.dtype != expected[name].dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid tensor schema/value: ' + name)


def complete_resnet_state(model, state, allowed_missing, torch):
    expected = model.state_dict()
    missing = set(expected) - set(state)
    if missing != set(allowed_missing) or set(state) - set(expected):
        raise ValueError('Only the predeclared missing BN counters may be added')
    modules = dict(model.named_modules())
    result = dict(state)
    for key in sorted(missing):
        if not key.endswith('.num_batches_tracked'):
            raise ValueError('Non-counter missing key')
        bn = modules[key.rsplit('.', 1)[0]]
        if (type(bn) is not torch.nn.BatchNorm2d or bn.momentum != 0.1
                or not bn.track_running_stats or expected[key].shape != torch.Size([])
                or expected[key].dtype != torch.int64):
            raise ValueError('Unsupported BN policy')
        result[key] = torch.zeros((), dtype=torch.int64)
    exact_state(result, expected, torch)
    return result


def load_resnet_candidate(torch, plan):
    from safetensors.numpy import load_file
    if digest(RESNET) != RESNET_SHA:
        raise ValueError('ResNet candidate bytes changed')
    state = {k: torch.from_numpy(v.copy()) for k, v in load_file(str(RESNET)).items()}
    evidence = read(ROOT / '35_pgsnet_resnet_type_audit/normalization_verification.json')
    if set(state) != set(evidence['per_key_value_sha256']):
        raise ValueError('Source key set changed')
    for key, value in state.items():
        if hashlib.sha256(value.numpy().tobytes()).hexdigest() != evidence['per_key_value_sha256'][key]:
            raise ValueError('Source tensor bytes changed')
    if len(plan['allowed_missing_resnet_counters']) != 20:
        raise ValueError('Expected 20 explicitly enumerated counters')
    return state


def strict_load(model, state, torch):
    exact_state(state, model.state_dict(), torch)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError('Strict load returned a mismatch')
    for key, value in model.state_dict().items():
        if not torch.equal(value, state[key]):
            raise ValueError('Post-load value mismatch: ' + key)


def pgsnet_module():
    # Load the unchanged three timm implementation files, not its package initializer.
    # Only the single upstream timm import is redirected in memory to these functions.
    package = types.ModuleType('pfe36_timm_layers')
    package.__path__ = [str(VENDOR)]
    sys.modules[package.__name__] = package
    module_from_file('pfe36_timm_layers.grid', VENDOR / 'grid.py')
    drop = module_from_file('pfe36_timm_layers.drop', VENDOR / 'drop.py')
    init = module_from_file('pfe36_timm_layers.weight_init', VENDOR / 'weight_init.py')
    path = UPSTREAM / 'model/pgsnet.py'
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == 'timm.models.layers']
    if (len(imports) != 1 or [(n.name, n.asname) for n in imports[0].names]
            != [('DropPath', None), ('trunc_normal_', None)]):
        raise ValueError('Upstream dependency import changed')
    tree.body.remove(imports[0])
    module = types.ModuleType('pfe36_pgsnet')
    module.__file__ = str(path)
    module.DropPath, module.trunc_normal_ = drop.DropPath, init.trunc_normal_
    sys.path.insert(0, str(UPSTREAM))
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong) for name in ('total_physical', 'available_physical',
        'total_pagefile', 'available_pagefile', 'total_virtual', 'available_virtual', 'extended')]


def available_gib():
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('Cannot verify memory availability')
    return status.available_physical / 1024**3


def memory_guard(minimum):
    available = available_gib()
    if available < minimum:
        raise RuntimeError(f'Insufficient spare RAM: {available:.3f} GiB < {minimum}')
    return available
