"""Synthetic backend diagnosis only; no relaxed acceptance or dataset access."""
import copy
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "15_pcod_training"))
import check_hardware as h
import torch


def differences(a, b):
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    delta = (a - b).abs()
    return {
        "max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
        "reference_abs_max": a.abs().max().item(),
        "relative_l2": (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(a).clamp_min(1e-30)).item(),
        "original_close": torch.allclose(a, b, atol=1e-5, rtol=1e-4),
        "original_mismatch_fraction": (delta > (1e-5 + 1e-4 * b.abs())).double().mean().item(),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    }


def evaluate(model, x, target):
    x = x.detach().clone().requires_grad_(True)
    logits = model(x)
    loss = h.t.segmentation_loss(logits, target)
    loss.backward()
    return {"logits": logits.detach().cpu(), "probability": logits.detach().sigmoid().cpu(),
            "loss": loss.detach().cpu(), "input_gradient": x.grad.detach().cpu()}


def compare(a, b):
    return {key: differences(a[key], b[key]) for key in a}


def main():
    torch.set_num_threads(1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = HERE / ("diagnostic_" + stamp + ".json")
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads(h.PROTOCOL.read_text())
    h.validate_protocol(protocol)
    device, device_name = h.t.choose_device(h.base_config("pfe_reference", 42, "hardware_smoke"))
    report = {"purpose": "diagnosis_not_acceptance", "created_utc": stamp,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "hardware_protocol_sha256": h.t.sha(h.PROTOCOL), "device": device_name,
              "test_data_read": False, "formal_training": False, "cases": []}
    for seed in (19, 71):
        for method in h.METHODS:
            model = h.t.build_model(method, base_channels=h.t.FORMAL_WIDTHS[method], seed=seed).eval()
            generator = torch.Generator().manual_seed(27 if seed == 19 else 83)
            x = torch.rand(2, 4, 64, 64, generator=generator)
            y = (torch.rand(2, 1, 64, 64, generator=generator) > .5).float()
            cpu = evaluate(model, x, y)
            with torch.backends.mkldnn.flags(enabled=False):
                plain = evaluate(copy.deepcopy(model), x, y)
            high = evaluate(copy.deepcopy(model).double(), x.double(), y.double())
            gpu_model = copy.deepcopy(model).to(device)
            gpu = evaluate(gpu_model, x.to(device), y.to(device))
            repeat = evaluate(gpu_model, x.to(device), y.to(device))
            case = {"method": method, "seed": seed, "size": 64,
                    "cpu32_vs_dml": compare(cpu, gpu), "cpu64_vs_cpu32": compare(high, cpu),
                    "cpu64_vs_cpu32_no_mkldnn": compare(high, plain),
                    "cpu64_vs_dml": compare(high, gpu), "dml_repeat": compare(gpu, repeat)}
            report["cases"].append(case)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"method": method, "seed": seed,
                              "cpu_dml_logits": case["cpu32_vs_dml"]["logits"],
                              "cpu64_dml_logits": case["cpu64_vs_dml"]["logits"]}), flush=True)
            del model, gpu_model, cpu, high, plain, gpu, repeat
            gc.collect()
    report["state"] = "completed"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "cases": len(report["cases"])}))


if __name__ == "__main__":
    main()
