"""Versioned hardware acceptance for corrected U-Net resampling, never formal."""
import argparse
import gc
import json
import os
from pathlib import Path
import diagnose as d
import resize_candidate as resize
import resize_factor2 as factor2
import torch

h, t = d.h, d.h.t
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "corrected_hardware_protocol.json"
REPORT = HERE / "corrected_hardware_verification.json"
STATUS = HERE / "corrected_hardware_status.json"
ORIGINAL_BUILD = t.build_model
ORIGINAL_VALIDATE = t.validate_config
SEMANTICS = "pytorch_half_pixel_bilinear_factor2_v1"
SOURCES = (Path(__file__), HERE / "diagnose.py", HERE / "resize_candidate.py", HERE / "resize_factor2.py")


def corrected_model(method, *, base_channels=32, seed=42):
    model = ORIGINAL_BUILD(method, base_channels=base_channels, seed=seed)
    if method != t.STANDARD_METHOD:
        model = resize.install_unet_candidate(model)
    return model


def corrected_identity(cfg):
    if cfg.get("unet_resampling") != SEMANTICS or cfg["purpose"] != "hardware_smoke":
        raise ValueError("Only registered corrected hardware checks are allowed")
    identity = ORIGINAL_VALIDATE(cfg)
    identity["source_sha256"].update({str(p.relative_to(t.WORK)): t.sha(p) for p in SOURCES})
    identity["resampling_semantics"] = SEMANTICS
    return identity


def configurations(method):
    configs = h.variants(method)
    for cfg in configs:
        cfg["run_id"] = cfg["run_id"].replace("acceptance_v1", "acceptance_v2_half_pixel")
        cfg["output_dir"] = str(t.HERE / "runs" / cfg["run_id"])
        cfg["unet_resampling"] = SEMANTICS
    return configs


def specification():
    old = json.loads(h.PROTOCOL.read_text())
    h.validate_protocol(old)
    items = [{"config": cfg, "identity": corrected_identity(cfg)} for m in h.METHODS for cfg in configurations(m)]
    return {"purpose": "corrected_resampling_hardware_acceptance_only", "methods": list(h.METHODS),
            "old_failed_protocol_sha256": t.sha(h.PROTOCOL), "old_sources": old["source_sha256"],
            "source_sha256": {str(p.relative_to(t.WORK)): t.sha(p) for p in SOURCES}, "configs": items,
            "backend_tolerances": old["backend_tolerances"], "resume_tolerances": old["resume_tolerances"],
            "test_data_read": False, "formal_runs_allowed": False, "old_failure_preserved": True}


def prepare():
    spec = specification()
    h.write_once(PROTOCOL, spec)
    print(json.dumps({"prepared": str(PROTOCOL), "sha256": t.sha(PROTOCOL), "configs": len(spec["configs"])}))


def run():
    spec = json.loads(PROTOCOL.read_text())
    if spec != specification():
        raise ValueError("Corrected hardware protocol/source identity changed")
    h.verify_dependency(json.loads(h.PROTOCOL.read_text()))
    if REPORT.exists() or STATUS.exists():
        raise FileExistsError("Prior corrected acceptance exists; inspect, do not overwrite/retry blindly")
    for method in h.METHODS:
        for cfg in configurations(method):
            if Path(cfg["output_dir"]).exists():
                raise FileExistsError(cfg["output_dir"])
    lock = HERE / "corrected_hardware.lock"
    with lock.open("x") as stream:
        json.dump({"pid": os.getpid(), "utc": t.utc_now(), "protocol_sha256": t.sha(PROTOCOL)}, stream)
    report = {"passed": False, "protocol_sha256": t.sha(PROTOCOL), "backend_parity": [], "updates": [],
              "gpu_resume": [], "formal_experiments_completed": 0, "test_evaluated": False}
    resize.bilinear_half_pixel = factor2.bilinear_half_pixel
    t.build_model, t.validate_config = corrected_model, corrected_identity
    try:
        torch.set_num_threads(1)
        device, name = t.choose_device(h.base_config("pfe_reference", 42, "hardware_smoke"))
        report["device"] = name
        report["dropout_replay"] = h.dropout_replay(device)
        for method in h.METHODS:
            t.write_json(STATUS, {"state": "running", "method": method, "phase": "backend_parity", "pid": os.getpid(), "utc": t.utc_now()})
            report["backend_parity"].append(h.backend_parity(method, device, spec["backend_tolerances"]))
        for method in h.METHODS:
            t.write_json(STATUS, {"state": "running", "method": method, "phase": "updates_and_resume", "pid": os.getpid(), "utc": t.utc_now()})
            update, full_cfg, split_cfg = configurations(method)
            t.run_config(update)
            report["updates"].append(h.audit_update(update))
            t.run_config(full_cfg)
            t.run_config(split_cfg, stop_after_epoch=1)
            t.run_config(split_cfg, resume=True)
            full = torch.load(Path(full_cfg["output_dir"]) / "latest.pt", map_location="cpu", weights_only=False)
            split = torch.load(Path(split_cfg["output_dir"]) / "latest.pt", map_location="cpu", weights_only=False)
            tolerance = spec["resume_tolerances"]
            a = h.compare_tree(full["model_state"], split["model_state"], tolerance)
            b = h.compare_tree(full["optimizer_state"], split["optimizer_state"], tolerance)
            if not torch.equal(full["rng"]["torch_cpu"], split["rng"]["torch_cpu"]):
                raise AssertionError("Resumed RNG differs")
            report["gpu_resume"].append({"method": method, "passed": True, "model_max_abs_difference": a,
                                         "optimizer_max_abs_difference": b, **tolerance})
            del full, split
            gc.collect()
        report["passed"] = True
        report["completed_utc"] = t.utc_now()
        t.write_json(REPORT, report)
        t.write_json(STATUS, {"state": "completed", "protocol_sha256": t.sha(PROTOCOL), "utc": t.utc_now()})
    except BaseException as error:
        report["error"] = str(error)
        report["failed_utc"] = t.utc_now()
        t.write_json(HERE / "corrected_hardware_failure.json", report)
        t.write_json(STATUS, {"state": "failed", "error": str(error), "automatic_retry": False, "utc": t.utc_now()})
        raise
    finally:
        t.build_model, t.validate_config = ORIGINAL_BUILD, ORIGINAL_VALIDATE
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run()
