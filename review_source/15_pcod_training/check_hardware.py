"""Frozen, serial PCOD device acceptance; never starts formal training."""

import argparse
import copy
import gc
import json
import os
from pathlib import Path

import torch
import train_pcod as t
from prepare_configs import base_config, write_once

METHODS = ("pfe_reference", "global_film_style", "wide_direct", t.STANDARD_METHOD)
PROTOCOL = t.HERE / "hardware_protocol.json"
REPORT = t.HERE / "hardware_verification.json"
STATUS = t.HERE / "hardware_status.json"
LOCK = t.HERE / "hardware.lock"


def variants(method):
    configurations = []
    for suffix in ("update", "resume_full", "resume_split"):
        cfg = base_config(method, 42, "hardware_smoke")
        cfg["run_id"] = f"pcod_acceptance_v1_{method}_{suffix}"
        cfg["output_dir"] = str(t.HERE / "runs" / cfg["run_id"])
        if suffix != "update":
            cfg["sample_limits"] = {"train": 8, "validation": 4}
        configurations.append(cfg)
    return configurations


def prepare():
    config_items, sources = [], {}
    for method in METHODS:
        for cfg in variants(method):
            identity = t.validate_config(cfg)
            sources.update(identity["source_sha256"])
            path = t.HERE / "configs" / (cfg["run_id"] + ".json")
            write_once(path, cfg)
            config_items.append({"path": str(path), "sha256": t.sha(path), "identity": identity})
    sources[str(Path(__file__).relative_to(t.WORK))] = t.sha(__file__)
    sources[str((t.HERE / "prepare_configs.py").relative_to(t.WORK))] = t.sha(t.HERE / "prepare_configs.py")
    sources[str((t.HERE / "run_after_registration.py").relative_to(t.WORK))] = t.sha(t.HERE / "run_after_registration.py")
    protocol = {
        "purpose": "hardware_acceptance_only", "methods": list(METHODS),
        "configs": config_items, "source_sha256": sources,
        "backend_tolerances": {"atol": 1e-5, "rtol": 1e-4},
        "resume_tolerances": {"atol": 1e-6, "rtol": 1e-4},
        "dropout_replay": "exact CPU RNG and dropout output equality required",
        "registration_protocol_sha256": t.sha(t.WORK / "13_registration_evaluation/protocol.json"),
        "requires_completed_development_and_registration": True,
        "formal_experiments": 0, "test_images_read": False,
    }
    write_once(PROTOCOL, protocol)
    print(json.dumps({"prepared": str(PROTOCOL), "sha256": t.sha(PROTOCOL), "configs": len(config_items), "gpu_started": False}))


def validate_protocol(protocol):
    if protocol["purpose"] != "hardware_acceptance_only" or protocol["methods"] != list(METHODS):
        raise ValueError("Unexpected hardware scope")
    for relative, expected in protocol["source_sha256"].items():
        path = (t.WORK / relative).resolve()
        if not path.is_relative_to(t.WORK.resolve()) or t.sha(path) != expected:
            raise ValueError(f"Hardware source changed: {relative}")
    expected_configs = [cfg for method in METHODS for cfg in variants(method)]
    if len(protocol["configs"]) != len(expected_configs):
        raise ValueError("Hardware configuration count changed")
    for item, expected in zip(protocol["configs"], expected_configs):
        if t.sha(item["path"]) != item["sha256"]:
            raise ValueError("Hardware configuration changed")
        cfg = json.loads(Path(item["path"]).read_text())
        if cfg != expected or t.validate_config(cfg) != item["identity"]:
            raise ValueError("Hardware run identity changed")
    if protocol["backend_tolerances"] != {"atol": 1e-5, "rtol": 1e-4} or protocol["resume_tolerances"] != {"atol": 1e-6, "rtol": 1e-4}:
        raise ValueError("Hardware numerical thresholds changed")


def verify_dependency(protocol):
    training = t.WORK / "12_training"
    queue = json.loads((training / "development_queue.json").read_text())
    status = json.loads((training / "rgbp_development_v1_status.json").read_text())
    run_ids = []
    for item in queue["configs"]:
        if t.sha(item["path"]) != item["sha256"]:
            raise ValueError("Development configuration changed")
        cfg = json.loads(Path(item["path"]).read_text())
        run_ids.append(cfg["run_id"])
        out = Path(cfg["output_dir"])
        record = json.loads((out / "status.json").read_text())
        if record["state"] != "completed" or record["completed_epoch"] != cfg["epochs"]:
            raise ValueError("Development is not complete")
        if t.sha(out / "latest.pt") != record["checkpoint_sha256"]:
            raise ValueError("Development checkpoint changed")
    if status["state"] != "completed" or status["completed"] != run_ids:
        raise ValueError("Development controller did not finish successfully")
    evaluation = t.WORK / "13_registration_evaluation"
    if t.sha(evaluation / "protocol.json") != protocol["registration_protocol_sha256"]:
        raise ValueError("Registration protocol changed")
    dependency = json.loads((evaluation / "dependency_status.json").read_text())
    status = json.loads((evaluation / "status.json").read_text())
    if dependency["state"] != "completed" or status["state"] != "completed" or status["conditions"] != 306:
        raise ValueError("Registration evaluation has not completed successfully")


def dropout_replay(device):
    module = torch.nn.Dropout(0.5).to(device).train()
    values = torch.ones(4, 256, 22, 22).to(device)
    torch.manual_seed(91)
    initial = torch.get_rng_state().clone()
    first = module(values).cpu()
    after = torch.get_rng_state().clone()
    torch.set_rng_state(initial)
    second = module(values).cpu()
    if not torch.equal(first, second) or not torch.equal(after, torch.get_rng_state()):
        raise AssertionError("DirectML Dropout cannot be replayed from saved CPU RNG state")
    retained = float(torch.count_nonzero(first)) / first.numel()
    if not 0.48 < retained < 0.52:
        raise AssertionError("Dropout is inactive or has an unexpected retention rate")
    return {"passed": True, "retained_fraction": retained, "rng_consumed": not torch.equal(initial, after)}


def backend_parity(method, device, tolerance):
    width = t.FORMAL_WIDTHS[method]
    cpu = t.build_model(method, base_channels=width, seed=19).eval()
    gpu = copy.deepcopy(cpu).to(device).eval()
    generator = torch.Generator().manual_seed(27)
    x = torch.rand(2, 4, 64, 64, generator=generator).requires_grad_(True)
    target = (torch.rand(2, 1, 64, 64, generator=generator) > 0.5).float()
    dx = x.detach().to(device).requires_grad_(True)
    a, b = cpu(x), gpu(dx)
    ca, cb = t.segmentation_loss(a, target), t.segmentation_loss(b, target.to(device))
    ca.backward()
    cb.backward()
    torch.testing.assert_close(a.detach(), b.detach().cpu(), **tolerance)
    torch.testing.assert_close(ca.detach(), cb.detach().cpu(), **tolerance)
    torch.testing.assert_close(x.grad, dx.grad.cpu(), **tolerance)
    return {"method": method, "logit_max_abs_difference": float((a.detach() - b.detach().cpu()).abs().max()),
            "loss_abs_difference": abs(float(ca.detach()) - float(cb.detach().cpu())),
            "input_gradient_max_abs_difference": float((x.grad - dx.grad.cpu()).abs().max()),
            "synthetic_size": 64, "full_model_width": width, **tolerance}


def compare_tree(first, second, tolerance):
    maximum = 0.0
    if isinstance(first, torch.Tensor):
        if first.is_floating_point():
            torch.testing.assert_close(first, second, **tolerance)
            maximum = float((first - second).abs().max()) if first.numel() else 0.0
        elif not torch.equal(first, second):
            raise AssertionError("Integer/RNG state mismatch")
    elif isinstance(first, dict):
        if first.keys() != second.keys():
            raise AssertionError("State keys differ")
        for key in first:
            maximum = max(maximum, compare_tree(first[key], second[key], tolerance))
    elif isinstance(first, (list, tuple)):
        if len(first) != len(second):
            raise AssertionError("State lengths differ")
        for a, b in zip(first, second):
            maximum = max(maximum, compare_tree(a, b, tolerance))
    elif first != second:
        raise AssertionError("Scalar state mismatch")
    return maximum


def audit_update(cfg):
    out = Path(cfg["output_dir"])
    state = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
    initial = t.build_model(cfg["method"], base_channels=cfg["base_channels"], seed=cfg["seed"])
    for name, value in state["model_state"].items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise AssertionError(f"Nonfinite parameter or buffer: {name}")
    changed = []
    for name, parameter in initial.named_parameters():
        value = state["model_state"][name]
        if not torch.isfinite(value).all():
            raise AssertionError(f"Nonfinite updated parameter: {name}")
        if not torch.equal(value, parameter):
            changed.append(name)
    steps = {int(entry["step"]) for entry in state["optimizer_state"]["state"].values()}
    if state["epoch"] != 2 or steps != {16} or not changed:
        raise AssertionError("Full-resolution updates did not complete as specified")
    initial.load_state_dict(state["model_state"], strict=True)
    initial.eval()
    generator = torch.Generator().manual_seed(401)
    x = torch.rand(2, 4, 64, 64, generator=generator).requires_grad_(True)
    target = (torch.rand(2, 1, 64, 64, generator=generator) > .5).float()
    t.segmentation_loss(initial(x), target).backward()
    auxiliary_gradient = float(x.grad[:, 3].abs().sum())
    if not torch.isfinite(x.grad).all() or auxiliary_gradient <= 0:
        raise AssertionError("DoLP input has no finite nonzero gradient after updates")
    return {"method": cfg["method"], "updated_parameter_tensors": len(changed), "optimizer_steps": 16,
            "checkpoint_sha256": t.sha(out / "latest.pt"), "finite": True,
            "dolp_input_gradient_l1_after_updates_cpu_check": auxiliary_gradient}


def run(protocol):
    validate_protocol(protocol)
    verify_dependency(protocol)
    with LOCK.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "started_utc": t.utc_now(), "protocol_sha256": t.sha(PROTOCOL)}))
    report = {"passed": False, "formal_experiments_completed": 0, "test_evaluated": False,
              "protocol_sha256": t.sha(PROTOCOL), "backend_parity": [], "updates": [], "gpu_resume": []}
    try:
        if REPORT.exists():
            raise FileExistsError("Acceptance report exists; inspect rather than overwrite")
        for method in METHODS:
            for cfg in variants(method):
                if Path(cfg["output_dir"]).exists():
                    raise FileExistsError(f"Acceptance run already exists: {cfg['run_id']}")
        torch.set_num_threads(1)
        device, name = t.choose_device(base_config("pfe_reference", 42, "hardware_smoke"))
        torch.zeros(1).to(device).cpu()
        report["device"] = name
        report["dropout_replay"] = dropout_replay(device)
        t.write_json(STATUS, {"state": "running", "pid": os.getpid(), "phase": "backend_parity", "updated_utc": t.utc_now()})
        for method in METHODS:
            report["backend_parity"].append(backend_parity(method, device, protocol["backend_tolerances"]))
            gc.collect()
        for method in METHODS:
            t.write_json(STATUS, {"state": "running", "pid": os.getpid(), "phase": "updates_and_resume",
                                  "method": method, "updated_utc": t.utc_now()})
            update, full_cfg, split_cfg = variants(method)
            t.run_config(update)
            report["updates"].append(audit_update(update))
            t.run_config(full_cfg)
            t.run_config(split_cfg, stop_after_epoch=1)
            t.run_config(split_cfg, resume=True)
            full = torch.load(Path(full_cfg["output_dir"]) / "latest.pt", map_location="cpu", weights_only=False)
            split = torch.load(Path(split_cfg["output_dir"]) / "latest.pt", map_location="cpu", weights_only=False)
            tolerance = protocol["resume_tolerances"]
            a = compare_tree(full["model_state"], split["model_state"], tolerance)
            b = compare_tree(full["optimizer_state"], split["optimizer_state"], tolerance)
            if not torch.equal(full["rng"]["torch_cpu"], split["rng"]["torch_cpu"]):
                raise AssertionError("Resumed sampler/dropout RNG differs")
            report["gpu_resume"].append({"method": method, "passed": True, "model_max_abs_difference": a,
                                         "optimizer_max_abs_difference": b, **tolerance})
            del full, split
            gc.collect()
        report["passed"] = True
        report["completed_utc"] = t.utc_now()
        t.write_json(REPORT, report)
        t.write_json(STATUS, {"state": "completed", "updated_utc": t.utc_now(), "protocol_sha256": t.sha(PROTOCOL)})
    except BaseException as error:
        report["error"] = str(error)
        report["failed_utc"] = t.utc_now()
        t.write_json(t.HERE / ("hardware_failure_" + report["failed_utc"].replace(":", "-") + ".json"), report)
        t.write_json(STATUS, {"state": "failed", "error": str(error), "automatic_retry": False, "updated_utc": t.utc_now()})
        raise
    finally:
        LOCK.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "validate", "run"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        protocol = json.loads(PROTOCOL.read_text())
        if args.action == "validate":
            validate_protocol(protocol)
            print(json.dumps({"protocol_valid": True, "gpu_started": False}))
        else:
            run(protocol)


if __name__ == "__main__":
    main()
