"""Bounded RGBP acceptance with explicit half-pixel decoder semantics."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
import types
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "12_training"))
import train_rgbp as t
import resize_candidate as resize
import resize_factor2 as fast

ORIGINAL_BUILD, ORIGINAL_VALIDATE = t.build_model, t.validate_config
PROTOCOL = HERE / "rgbp_corrected_hardware_protocol.json"
REPORT = HERE / "rgbp_corrected_hardware_verification.json"
STATUS = HERE / "rgbp_corrected_hardware_status.json"
SEMANTICS = "pytorch_half_pixel_bilinear_factor2_v1"
SOURCES = (Path(__file__), HERE / "resize_candidate.py", HERE / "resize_factor2.py")


def corrected_model(method, *, base_channels=32, seed=42):
    model = ORIGINAL_BUILD(method, base_channels=base_channels, seed=seed)
    count = 0
    for module in model.modules():
        if type(module).__name__ == "Up" and type(module).__module__ == "reference_models.unet":
            module.forward = types.MethodType(resize.portable_up_forward, module)
            count += 1
    if count != 4:
        raise ValueError("Unexpected RGBP reference model")
    return model


def identity(cfg):
    if cfg.get("unet_resampling") != SEMANTICS or cfg["purpose"] != "hardware_smoke":
        raise ValueError("Only registered corrected hardware checks are allowed")
    result = ORIGINAL_VALIDATE(cfg)
    result["source_sha256"].update({str(p.relative_to(ROOT)): t.sha(p) for p in SOURCES})
    result["resampling_semantics"] = SEMANTICS
    return result


def configs():
    queue = json.loads((ROOT / "12_training/development_queue.json").read_text())
    result = []
    for entry in queue["configs"]:
        if t.sha(entry["path"]) != entry["sha256"]:
            raise ValueError("Old development config changed")
        parent = json.loads(Path(entry["path"]).read_text())
        for suffix in ("update", "resume_full", "resume_split"):
            cfg = copy.deepcopy(parent)
            cfg.update({"run_id": f"rgbp_acceptance_v2_half_pixel_{parent['method']}_{suffix}",
                        "purpose": "hardware_smoke", "seed": 42, "epochs": 2,
                        "unet_resampling": SEMANTICS,
                        "sample_limits": {"train": 32 if suffix == "update" else 8,
                                          "validation": 8 if suffix == "update" else 4}})
            cfg["output_dir"] = str(t.HERE / "runs" / cfg["run_id"])
            result.append(cfg)
    return result


def spec():
    return {"purpose": "rgbp_corrected_hardware_only", "test_read": False,
            "source_sha256": {str(p.relative_to(ROOT)): t.sha(p) for p in SOURCES},
            "configs": [{"config": c, "identity": identity(c)} for c in configs()],
            "parity_tolerance": {"atol": 1e-5, "rtol": 1e-4},
            "resume_tolerance": {"atol": 1e-6, "rtol": 1e-4}}


def compare_tree(a, b, tolerance):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, **tolerance) if a.is_floating_point() else torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError("State keys differ")
        for key in a:
            compare_tree(a[key], b[key], tolerance)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError("Sequence sizes differ")
        for x, y in zip(a, b):
            compare_tree(x, y, tolerance)
    elif a != b:
        raise AssertionError("State scalars differ")


def run():
    expected = json.loads(PROTOCOL.read_text())
    if expected != spec():
        raise ValueError("Frozen hardware contract changed")
    if REPORT.exists() or STATUS.exists():
        raise FileExistsError("Check already attempted; inspect before retry")
    if any(Path(c["output_dir"]).exists() for c in configs()):
        raise FileExistsError("Hardware output exists")
    lock = HERE / "rgbp_corrected_hardware.lock"
    with lock.open("x") as stream:
        json.dump({"pid": os.getpid(), "utc": t.utc_now()}, stream)
    resize.bilinear_half_pixel = fast.bilinear_half_pixel
    t.build_model, t.validate_config = corrected_model, identity
    report = {"passed": False, "protocol_sha256": t.sha(PROTOCOL), "checks": [], "formal_runs": 0, "test_read": False}
    try:
        torch.set_num_threads(1)
        all_configs = configs()
        for start in range(0, len(all_configs), 3):
            update, full_cfg, split_cfg = all_configs[start:start + 3]
            method = update["method"]
            t.write_json(STATUS, {"state": "running", "method": method, "pid": os.getpid(), "utc": t.utc_now()})
            device, name = t.choose_device(update)
            report["device"] = name
            cpu = ORIGINAL_BUILD(method, base_channels=32, seed=19).eval()
            gpu = corrected_model(method, base_channels=32, seed=19).to(device).eval()
            generator = torch.Generator().manual_seed(27)
            x = torch.rand(2, 3 if method == "rgb_only" else 6, 64, 64, generator=generator).requires_grad_(True)
            y = (torch.rand(2, 1, 64, 64, generator=generator) > .5).float()
            dx = x.detach().to(device).requires_grad_(True)
            a, b = cpu(x), gpu(dx)
            la, lb = t.segmentation_loss(a, y), t.segmentation_loss(b, y.to(device))
            la.backward()
            lb.backward()
            for ca, cb in ((a, b.cpu()), (la, lb.cpu()), (x.grad, dx.grad.cpu())):
                torch.testing.assert_close(ca, cb, **expected["parity_tolerance"])
            logit_diff = float((a.detach() - b.detach().cpu()).abs().max())
            del cpu, gpu, a, b, la, lb, x, dx
            t.run_config(update)
            state = torch.load(Path(update["output_dir"]) / "latest.pt", map_location="cpu", weights_only=False)
            if state["epoch"] != 2 or {int(v["step"]) for v in state["optimizer_state"]["state"].values()} != {16}:
                raise AssertionError("Expected 16 update steps")
            if not all(bool(torch.isfinite(v).all()) for v in state["model_state"].values()):
                raise AssertionError("Nonfinite state")
            del state
            t.run_config(full_cfg)
            t.run_config(split_cfg, stop_after_epoch=1)
            t.run_config(split_cfg, resume=True)
            a = torch.load(Path(full_cfg["output_dir"]) / "latest.pt", map_location="cpu", weights_only=False)
            b = torch.load(Path(split_cfg["output_dir"]) / "latest.pt", map_location="cpu", weights_only=False)
            for key in ("model_state", "optimizer_state"):
                compare_tree(a[key], b[key], expected["resume_tolerance"])
            if not torch.equal(a["rng"]["torch_cpu"], b["rng"]["torch_cpu"]):
                raise AssertionError("RNG differs after resume")
            report["checks"].append({"method": method, "passed": True, "logit_max_abs_difference": logit_diff,
                                     "steps": 16, "gpu_resume": True, "source_semantics": SEMANTICS})
            del a, b
            gc.collect()
        report.update({"passed": True, "completed_utc": t.utc_now()})
        t.write_json(REPORT, report)
        t.write_json(STATUS, {"state": "completed", "utc": t.utc_now(), "protocol_sha256": t.sha(PROTOCOL)})
    except BaseException as error:
        report["error"] = str(error)
        t.write_json(HERE / "rgbp_corrected_hardware_failure.json", report)
        t.write_json(STATUS, {"state": "failed", "error": str(error), "automatic_retry": False, "utc": t.utc_now()})
        raise
    finally:
        t.build_model, t.validate_config = ORIGINAL_BUILD, ORIGINAL_VALIDATE
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    if args.action == "prepare":
        data = spec()
        if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != data:
            raise FileExistsError(PROTOCOL)
        t.write_json(PROTOCOL, data)
        print(json.dumps({"prepared": str(PROTOCOL), "sha256": t.sha(PROTOCOL)}))
    else:
        run()
