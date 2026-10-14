"""Pinned CPU-runtime gate for external weights, with no unsafe fallback."""

import json
from pathlib import Path
import re
import sys

from prepare_runtime import digest, save, HERE, ROOT, ENV


def supported_version(version):
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\+cpu)?", version)
    return bool(match and tuple(map(int, match.groups())) >= (2, 10, 0))


def schema_difference(state, expected):
    common = sorted(set(state) & set(expected))
    result = {
        "missing": sorted(set(expected) - set(state)),
        "unexpected": sorted(set(state) - set(expected)),
        "shape_mismatches": [k for k in common if list(state[k].shape) != expected[k]["shape"]],
        "dtype_mismatches": [k for k in common if str(state[k].dtype) != expected[k]["dtype"]],
    }
    result["strict_compatible"] = not any(result.values())
    return result


def require_runtime():
    import torch
    if not supported_version(torch.__version__):
        raise RuntimeError("External weights blocked: reviewed deserialization fixes require >=2.10.0")
    ready = json.loads((HERE / "runtime_ready.json").read_text())
    if Path(sys.prefix).resolve() != ENV.resolve() or torch.__version__ != ready["torch"]:
        raise RuntimeError("Use only the pinned separate CPU runtime")
    if torch.version.cuda is not None or not Path(torch.__file__).resolve().is_relative_to(ENV.resolve()):
        raise RuntimeError("Unexpected torch installation")
    return torch


def main():
    torch = require_runtime()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = ROOT / "30_pgsnet_loss_preparation"
    sys.path.insert(0, str(base))
    import checkpoint_inspection as inspector
    schema_file = HERE / "expected_schemas.json"
    schemas = json.loads(schema_file.read_text())["schemas"]
    out = HERE / "external_backbones_proxy"
    downloads = json.loads((out / "download_manifest.json").read_text())
    results = []
    for item in downloads:
        row = {"name": item["name"], "download_status": item["status"],
               "schema_sha256": digest(schema_file), "runtime": torch.__version__,
               "model_parameters_loaded": False, "training_started": False}
        if item["status"] != "downloaded_not_loaded":
            row.update(status="not_attempted", reason="No accepted complete download")
            results.append(row)
            continue
        try:
            state, details = inspector.inspect_checkpoint(out / item["name"], expected_sha256=item["sha256"])
            comparison = schema_difference(state, schemas[item["name"]])
            row.update(inspection=details, comparison=comparison,
                       status="schema_compatible" if comparison["strict_compatible"] else "schema_mismatch")
            # Stop at audit; never strip keys, reshape, cast, or call model.load_state_dict.
            del state
        except Exception as error:
            row.update(status="rejected", error=f"{type(error).__name__}: {error}")
        results.append(row)
        print(json.dumps(row), flush=True)
    save(HERE / "checkpoint_inspection.json", {
        "results": results, "guard_sha256": digest(Path(__file__)),
        "legacy_helper_sha256": digest(base / "checkpoint_inspection.py"),
        "security_guarantee": False, "no_unsafe_fallback": True,
        "active_environment_used_to_load_external_weights": False,
    })


if __name__ == "__main__":
    main()
