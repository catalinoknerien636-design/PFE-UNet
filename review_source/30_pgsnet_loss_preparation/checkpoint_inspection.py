"""Restricted CPU tensor inspection only; no unsafe-load or model-load fallback.

This does not authenticate a checkpoint or guarantee sandbox security. External
files still require verified provenance and a current, isolated loading runtime.
"""

from collections import OrderedDict
import hashlib
import io
from pathlib import Path, PurePosixPath
import re
import zipfile

import torch


def inspect_checkpoint(path, *, expected_sha256, max_file_bytes=512 * 1024**2,
                       max_archive_bytes=768 * 1024**2, locally_generated_test_fixture=False):
    version = re.match(r"^(\d+)\.(\d+)\.(\d+)", torch.__version__)
    if not version or tuple(map(int, version.groups())) < (2, 6, 0):
        if not locally_generated_test_fixture:
            raise RuntimeError("External checkpoints blocked: CVE-2025-32434 affects this runtime")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("An explicit lowercase SHA256 is required")
    if max_file_bytes <= 0 or max_archive_bytes <= 0:
        raise ValueError("Positive byte limits required")
    path = Path(path)
    with path.open("rb") as stream:
        payload = stream.read(max_file_bytes + 1)
    if len(payload) > max_file_bytes:
        raise ValueError("Checkpoint exceeds file byte limit")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Checkpoint SHA256 mismatch")
    buffer = io.BytesIO(payload)
    if not zipfile.is_zipfile(buffer):
        raise ValueError("Legacy/non-ZIP checkpoint needs separate format review")
    with zipfile.ZipFile(buffer) as archive:
        members = archive.infolist()
        if len(members) > 10000 or sum(x.file_size for x in members) > max_archive_bytes:
            raise ValueError("Archive exceeds declared resource limits")
        for member in members:
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in member.filename or member.flag_bits & 1:
                raise ValueError("Unexpected archive member")
    buffer.seek(0)
    # Never retry weights_only=False or allowlist external Python globals.
    payload_object = torch.load(buffer, weights_only=True, map_location="cpu")
    if type(payload_object) not in (dict, OrderedDict):
        raise ValueError("Checkpoint is not a tensor state mapping")
    state = payload_object
    container = "root"
    if not all(isinstance(k, str) and type(v) is torch.Tensor for k, v in state.items()):
        nested = [k for k in ("state_dict", "model") if k in state and type(state[k]) in (dict, OrderedDict)]
        if len(nested) != 1:
            raise ValueError("No unambiguous supported tensor state mapping")
        container = nested[0]
        state = state[container]
    if not state or not all(isinstance(k, str) and k and type(v) is torch.Tensor for k, v in state.items()):
        raise ValueError("Expected a nonempty string-to-plain-tensor mapping")
    tensor_bytes = 0
    for value in state.values():
        if value.device.type != "cpu" or value.layout != torch.strided or value.is_quantized:
            raise ValueError("Only dense CPU tensors supported")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite checkpoint tensor")
        tensor_bytes += value.numel() * value.element_size()
    if tensor_bytes > max_archive_bytes:
        raise ValueError("Tensor payload exceeds resource limit")
    return state, {"sha256": digest, "bytes": len(payload), "container": container,
                   "keys": len(state), "tensor_bytes": tensor_bytes,
                   "weights_only": True, "device": "cpu", "model_modified": False}


def compare_schema(state, expected):
    """Return all mismatches without removing prefixes or loading any parameter."""
    actual_keys, expected_keys = set(state), set(expected)
    common = sorted(actual_keys & expected_keys)
    report = {
        "missing": sorted(expected_keys - actual_keys),
        "unexpected": sorted(actual_keys - expected_keys),
        "shape_mismatches": [k for k in common if state[k].shape != expected[k].shape],
        "dtype_mismatches": [k for k in common if state[k].dtype != expected[k].dtype],
    }
    report["strict_compatible"] = not any(report.values())
    return report
