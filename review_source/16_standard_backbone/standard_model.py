"""Pinned SMP DeepLabV3+ / ResNet-50 candidate; no implicit weight downloads."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
MODEL_SPEC = {
    "encoder_name": "resnet50",
    "encoder_depth": 5,
    "encoder_weights": None,
    "encoder_output_stride": 16,
    "decoder_channels": 256,
    "decoder_atrous_rates": [6, 12, 18],
    "decoder_aspp_separable": True,
    "decoder_aspp_dropout": 0.5,
    "in_channels": 4,
    "classes": 1,
    "activation": None,
    "upsampling": 4,
    "aux_params": None,
}


def verify_environment():
    lock = json.loads((ROOT / "environment_lock.json").read_text(encoding="utf-8"))
    for relative, expected in lock["files_sha256"].items():
        path = (VENDOR / relative).resolve()
        if not path.is_relative_to(VENDOR.resolve()) or not path.is_file():
            raise RuntimeError(f"Missing/invalid environment file: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Environment file changed: {relative}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(ROOT / "cache" / "huggingface")
    os.environ["TORCH_HOME"] = str(ROOT / "cache" / "torch")
    sys.dont_write_bytecode = True
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    versions = {}
    for name, expected in (
        ("segmentation_models_pytorch", "0.5.0"), ("timm", "1.0.15"),
        ("huggingface_hub", "0.30.2"), ("safetensors", "0.5.3"),
    ):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(VENDOR.resolve()):
            raise RuntimeError(f"Unexpected import location: {name}: {module.__file__}")
        if module.__version__ != expected:
            raise RuntimeError(f"Unexpected package version: {name}")
        versions[name] = module.__version__
    return versions


def build_model(seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("Nonnegative integer seed required")
    verify_environment()
    import torch
    import segmentation_models_pytorch as smp

    torch.manual_seed(seed)
    return smp.DeepLabV3Plus(**MODEL_SPEC)
