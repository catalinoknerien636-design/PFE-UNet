"""Fresh, equal-budget RGBP development with corrected decoder resampling."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "20_pcod_backend_diagnostics"))
import check_rgbp_corrected as backend

t = backend.t
QUEUE = HERE / "development_queue.json"
METHODS = tuple(t.METHODS)
SOURCES = (Path(__file__), HERE / "run_queue.py", Path(backend.__file__),
           ROOT / "20_pcod_backend_diagnostics/resize_candidate.py",
           ROOT / "20_pcod_backend_diagnostics/resize_factor2.py")


def configurations():
    old_queue = json.loads((ROOT / "12_training/development_queue.json").read_text())
    result = []
    for entry in old_queue["configs"]:
        if t.sha(entry["path"]) != entry["sha256"]:
            raise ValueError("Historical development config changed")
        cfg = json.loads(Path(entry["path"]).read_text())
        cfg["run_id"] = f"rgbp_dev_v3_half_pixel_{cfg['method']}_seed17"
        cfg["output_dir"] = str(t.HERE / "runs" / cfg["run_id"])
        cfg["epochs"] = 50
        cfg["unet_resampling"] = backend.SEMANTICS
        cfg["initialization"] = "from_scratch_not_v1_or_v2_checkpoint"
        result.append(cfg)
    if tuple(c["method"] for c in result) != METHODS:
        raise ValueError("Expected exactly five ordered methods")
    return result


def identity(cfg):
    expected = {c["method"]: c for c in configurations()}
    if cfg.get("method") not in expected or cfg != expected[cfg["method"]]:
        raise ValueError("Development config differs from common frozen design")
    if cfg["purpose"] != "development" or cfg["seed"] != 17 or cfg["sample_limits"] != {"train": None, "validation": None}:
        raise ValueError("Not full-pool seed17 development")
    value = backend.ORIGINAL_VALIDATE(cfg)
    value["source_sha256"].update({str(p.relative_to(ROOT)): t.sha(p) for p in SOURCES})
    value["resampling_semantics"] = backend.SEMANTICS
    return value


def verify_hardware():
    protocol = backend.PROTOCOL
    report = json.loads(backend.REPORT.read_text())
    state = json.loads(backend.STATUS.read_text())
    checks = report["checks"]
    if (not report["passed"] or state["state"] != "completed"
            or report["protocol_sha256"] != t.sha(protocol) or state["protocol_sha256"] != t.sha(protocol)
            or len(checks) != 5 or {r["method"] for r in checks} != set(METHODS)
            or not all(r["passed"] and r["gpu_resume"] and r["steps"] == 16 for r in checks)):
        raise ValueError("Five-method corrected hardware gate not satisfied")
    if json.loads(protocol.read_text()) != backend.spec():
        raise ValueError("Corrected hardware source/config changed")
    return {"protocol_sha256": t.sha(protocol), "report_sha256": t.sha(backend.REPORT)}


def validate_queue(queue):
    if (queue["purpose"] != "development" or queue["formal_result"] is not False
            or queue["test_evaluation"] is not False or queue["from_scratch"] is not True):
        raise ValueError("Invalid queue scope")
    if queue["hardware_gate"] != verify_hardware():
        raise ValueError("Hardware evidence changed")
    expected = configurations()
    if len(queue["configs"]) != len(expected):
        raise ValueError("Queue matrix changed")
    for item, cfg in zip(queue["configs"], expected):
        if t.sha(item["path"]) != item["sha256"] or json.loads(Path(item["path"]).read_text()) != cfg:
            raise ValueError("Queued config changed")
        if identity(cfg) != item["identity"]:
            raise ValueError("Queued source/runtime/manifest changed")


def prepare():
    gate = verify_hardware()
    items = []
    for cfg in configurations():
        item_identity = identity(cfg)
        path = HERE / "configs" / (cfg["run_id"] + ".json")
        if path.exists() and json.loads(path.read_text()) != cfg:
            raise FileExistsError(path)
        t.write_json(path, cfg)
        items.append({"path": str(path), "sha256": t.sha(path), "identity": item_identity})
    queue = {"queue_id": "rgbp_development_v3_half_pixel_budget50", "purpose": "development",
             "formal_result": False, "test_evaluation": False, "from_scratch": True,
             "hardware_gate": gate, "configs": items,
             "reason": "Native DirectML resize coordinate mismatch; no continuation across changed operator semantics"}
    if QUEUE.exists() and json.loads(QUEUE.read_text()) != queue:
        raise FileExistsError("Frozen queue differs")
    t.write_json(QUEUE, queue)
    print(json.dumps({"queue": str(QUEUE), "sha256": t.sha(QUEUE), "formal_runs": 0}))


def run(path, resume):
    queue = json.loads(QUEUE.read_text())
    validate_queue(queue)
    if path.resolve() not in {Path(r["path"]).resolve() for r in queue["configs"]}:
        raise ValueError("Config is not in the fixed development queue")
    cfg = json.loads(path.read_text())
    backend.resize.bilinear_half_pixel = backend.fast.bilinear_half_pixel
    t.build_model, t.validate_config = backend.corrected_model, identity
    try:
        return t.run_config(cfg, resume=resume)
    finally:
        t.build_model, t.validate_config = backend.ORIGINAL_BUILD, backend.ORIGINAL_VALIDATE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run(args.config, args.resume)
