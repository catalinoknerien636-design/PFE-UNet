"""PCOD training with archived split and loss semantics; no test evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
from pcod_models import METHODS, FORMAL_WIDTHS, REFERENCE, STANDARD_METHOD, standard_module, build_model, segmentation_loss
from scripts.extension_dataset import ExtensionPCODDataset
PROJECT = Path('<LOCAL_HOST_PATH_01>')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def save_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def make_loader(dataset, cfg, epoch, train):
    if cfg["horizontal_flip_probability"] != 0:
        raise ValueError("Historical PCOD uses stored derived frames, without this online flip")
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=train,
                      num_workers=0, drop_last=False)


class Metrics:
    def __init__(self):
        self.rows = []

    def update(self, logits, masks, stems):
        probabilities = torch.sigmoid(logits.detach()).float().cpu()
        masks = masks.detach().float().cpu()
        if probabilities.shape != masks.shape or not torch.isfinite(probabilities).all():
            raise ValueError("Invalid validation output")
        if len(stems) != len(masks):
            raise ValueError("Stem count differs from batch size")
        if not ((masks == 0) | (masks == 1)).all():
            raise ValueError("Binary masks required")
        for probability, target, stem in zip(probabilities, masks, stems):
            prediction, truth = probability > 0.5, target > 0.5
            self.rows.append({"stem": stem,
                              "tp": int((prediction & truth).sum()),
                              "fp": int((prediction & ~truth).sum()),
                              "fn": int((~prediction & truth).sum()),
                              "tn": int((~prediction & ~truth).sum()),
                              "mae": float((probability - target).abs().mean())})

    def finalize(self):
        if not self.rows:
            raise ValueError("Empty evaluation")
        if len({row["stem"] for row in self.rows}) != len(self.rows):
            raise ValueError("Duplicate evaluated sample")
        tp, fp, fn, tn = (sum(row[key] for row in self.rows) for key in ("tp", "fp", "fn", "tn"))
        eps = 1e-7
        return {"iou": tp / (tp + fp + fn + eps),
                "f1": 2 * tp / (2 * tp + fp + fn + eps),
                "precision": tp / (tp + fp + eps), "recall": tp / (tp + fn + eps),
                "image_mae": sum(row["mae"] for row in self.rows) / len(self.rows),
                "images": len(self.rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def choose_device(cfg):
    if cfg["device"] == "cpu":
        return torch.device("cpu"), "CPU"
    if cfg["device"] != "directml":
        raise ValueError("This tested runner supports CPU and DirectML only")
    import torch_directml
    index = cfg["device_index"]
    name = torch_directml.device_name(index).rstrip("\x00")
    if cfg["expected_device_name"] != name:
        raise ValueError(f"Device identity mismatch: {name}")
    return torch_directml.device(index), name


def run_epoch(model, loader, device, optimizer=None, progress=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, images = 0.0, 0
    metrics = None if training else Metrics()
    started = time.perf_counter()
    with torch.set_grad_enabled(training):
        for step, batch in enumerate(loader, 1):
            image, mask = batch["image"].to(device), batch["mask"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = segmentation_loss(logits, mask)
            scalar = float(loss.detach().cpu())
            if not math.isfinite(scalar):
                raise FloatingPointError("Nonfinite loss; do not skip or hide the batch")
            if training:
                loss.backward()
                optimizer.step()
            else:
                metrics.update(logits, mask, batch["stem"])
            count = len(batch["stem"])
            loss_sum += scalar * count
            images += count
            if progress and (step == 1 or step % 25 == 0 or step == len(loader)):
                progress({"phase": "train" if training else "validation", "step": step,
                          "steps": len(loader), "images": images, "last_loss": scalar})
    if not images:
        raise ValueError("Empty epoch")
    return {"loss": loss_sum / images, "images": images,
            "seconds": time.perf_counter()-started,
            "metrics": metrics.finalize() if metrics else None}, metrics


def export_history(out, history):
    path = out / "history.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def fit(cfg, train_set, validation_set, identity, *, resume=False, stop_after_epoch=None):
    out = Path(cfg["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "latest.pt"
    if not resume and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a run: {out}")
    if resume and not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint missing: {path}")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device, device_name = choose_device(cfg)
    model = build_model(cfg["method"], base_channels=cfg["base_channels"], seed=cfg["seed"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"], foreach=False)
    start, history, best_iou, best_epoch, best_state, best_rows = 1, [], -1., 0, None, []
    if resume:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["identity"] != identity:
            raise ValueError("Config, input or source identity changed; resume rejected")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start = checkpoint["epoch"] + 1
        history, best_iou, best_epoch = checkpoint["history"], checkpoint["best_iou"], checkpoint["best_epoch"]
        best_state, best_rows = checkpoint["best_model_state"], checkpoint["best_validation_rows"]
        torch.set_rng_state(checkpoint["rng"]["torch_cpu"])
        random.setstate(checkpoint["rng"]["python"])
        np.random.set_state(checkpoint["rng"]["numpy"])
    write_json(out / "config.json", cfg)
    write_json(out / "identity.json", identity)
    last_epoch = min(cfg["epochs"], stop_after_epoch) if stop_after_epoch is not None else cfg["epochs"]
    if last_epoch < start - 1:
        raise ValueError("Stop epoch precedes completed checkpoint")
    completed_epoch = start - 1
    for epoch in range(start, last_epoch + 1):
        def progress(record):
            write_json(out / "status.json", {"state": "running", "pid": os.getpid(),
                       "updated_utc": utc_now(), "run_id": cfg["run_id"], "epoch": epoch, **record})
        train, _ = run_epoch(model, make_loader(train_set, cfg, epoch, True), device, optimizer, progress)
        validation, measurements = run_epoch(model, make_loader(validation_set, cfg, epoch, False), device,
                                             progress=progress)
        row = {"epoch": epoch, "train_loss": train["loss"], "val_loss": validation["loss"],
               **{"val_"+key: value for key, value in validation["metrics"].items()},
               "train_seconds": train["seconds"], "validation_seconds": validation["seconds"],
               "train_images": train["images"]}
        history.append(row)
        if row["val_iou"] > best_iou:
            best_iou, best_epoch = row["val_iou"], epoch
            best_state, best_rows = cpu_tree(model.state_dict()), measurements.rows
        checkpoint = {"epoch": epoch, "identity": identity, "model_state": cpu_tree(model.state_dict()),
                      "optimizer_state": cpu_tree(optimizer.state_dict()), "history": history,
                      "best_iou": best_iou, "best_epoch": best_epoch,
                      "best_model_state": best_state, "best_validation_rows": best_rows,
                      "rng": {"torch_cpu": torch.get_rng_state(), "python": random.getstate(),
                              "numpy": np.random.get_state()}, "device_name": device_name}
        # Latest owns current and best state, even if CSV/best exports are interrupted.
        save_checkpoint(path, checkpoint)
        export_history(out, history)
        completed_epoch = epoch
        print(json.dumps({"run_id": cfg["run_id"], **row, "best_epoch": best_epoch}), flush=True)
    if history:
        export_history(out, history)
        save_checkpoint(out / "best.pt", {"identity": identity, "epoch": best_epoch,
                        "model_state": best_state, "validation_iou": best_iou})
        write_json(out / "best_validation_per_image.json", best_rows)
    status = {"state": "completed" if completed_epoch == cfg["epochs"] else "stopped_at_epoch_boundary",
              "run_id": cfg["run_id"], "completed_epoch": completed_epoch,
              "best_epoch": best_epoch, "best_validation_iou": best_iou,
              "updated_utc": utc_now(), "device": device_name, "test_evaluated": False,
              "purpose": cfg["purpose"], "checkpoint_sha256": sha(path) if path.is_file() else None}
    write_json(out / "status.json", status)
    return status


def validate_config(cfg):
    required = {"run_id", "purpose", "method", "seed", "image_size", "base_channels", "batch_size", "epochs",
                "learning_rate", "weight_decay", "horizontal_flip_probability", "device", "device_index",
                "expected_device_name", "dataset_root", "train_manifest", "validation_manifest", "manifest_sha256",
                "output_dir", "training_authorized", "protocol_frozen", "sample_limits", "loss", "sampling"}
    if not required <= cfg.keys():
        raise ValueError(f"Missing config keys: {required - cfg.keys()}")
    if cfg["purpose"] not in {"hardware_smoke", "development", "formal"} or cfg["method"] not in METHODS:
        raise ValueError("Unknown purpose or method")
    if cfg["training_authorized"] is not True:
        raise ValueError("Training is not authorized")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("Invalid seed")
    if type(cfg["device_index"]) is not int or cfg["device_index"] < 0:
        raise ValueError("Invalid device index")
    if set(cfg["sample_limits"]) != {"train", "validation"}:
        raise ValueError("Unexpected sample-limit keys")
    for key in ("epochs", "batch_size", "image_size"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"Invalid positive integer: {key}")
    if cfg["method"] == STANDARD_METHOD:
        if cfg["base_channels"] is not None or cfg.get("model_spec") != standard_module().MODEL_SPEC:
            raise ValueError("Standard-backbone architecture differs from registered specification")
        if cfg["batch_size"] < 2:
            raise ValueError("ASPP pooling BatchNorm requires at least two training images")
    elif type(cfg["base_channels"]) is not int or cfg["base_channels"] < 1:
        raise ValueError("Invalid base width")
    if cfg["image_size"] < 32 or cfg["horizontal_flip_probability"] != 0:
        raise ValueError("Invalid input size or augmentation")
    if cfg["loss"] != "historical_bce_dice_v1" or cfg["sampling"] != "legacy_global_torch_rng":
        raise ValueError("Historical training contract changed")
    if Path(cfg["dataset_root"]).resolve() != (PROJECT / "datasets/PCOD_1200/full-dataset").resolve():
        raise ValueError("Unexpected PCOD dataset root")
    if (not math.isfinite(cfg["learning_rate"]) or not math.isfinite(cfg["weight_decay"])
            or cfg["learning_rate"] <= 0 or cfg["weight_decay"] < 0):
        raise ValueError("Invalid optimizer settings")
    if cfg["purpose"] == "hardware_smoke":
        if cfg["epochs"] > 5 or any(type(n) is not int or not 1 <= n <= 64 for n in cfg["sample_limits"].values()):
            raise ValueError("Hardware-smoke scope must remain bounded")
    if cfg["purpose"] == "formal":
        if cfg["protocol_frozen"] is not True or cfg["sample_limits"] != {"train": None, "validation": None}:
            raise ValueError("Formal runs require a frozen protocol and full manifests")
        if cfg["image_size"] != 352 or cfg["base_channels"] != FORMAL_WIDTHS[cfg["method"]]:
            raise ValueError("Formal width/resolution differ from the specified main protocol")
        if (cfg["epochs"], cfg["batch_size"], cfg["learning_rate"], cfg["weight_decay"]) != (50, 4, 1e-4, 1e-4):
            raise ValueError("Formal training budget differs from archived PCOD runs")
        if cfg["seed"] not in (42, 2026, 3407, 7819, 12011):
            raise ValueError("Unregistered formal seed")
        if cfg["method"] == STANDARD_METHOD and cfg["seed"] not in (42, 2026, 3407):
            raise ValueError("Unregistered standard-backbone seed")
    for part in ("train", "validation"):
        path = Path(cfg[part + "_manifest"])
        if sha(path) != cfg["manifest_sha256"][part]:
            raise ValueError(f"Manifest hash mismatch: {part}")
    with Path(cfg["train_manifest"]).open(encoding="utf-8-sig", newline="") as stream:
        train = list(csv.DictReader(stream))
    with Path(cfg["validation_manifest"]).open(encoding="utf-8-sig", newline="") as stream:
        val = list(csv.DictReader(stream))
    if not train or not val or any(not row.get("stem") or not row.get("original_stem") for row in train + val):
        raise ValueError("Missing PCOD stem/group")
    if len({row["stem"] for row in train + val}) != len(train) + len(val):
        raise ValueError("Duplicate or overlapping train/validation members")
    if {r["stem"] for r in train} & {r["stem"] for r in val}:
        raise ValueError("Overlapping train/validation members")
    if {r["original_stem"] for r in train} & {r["original_stem"] for r in val}:
        raise ValueError("Overlapping source groups")
    if cfg["method"] == STANDARD_METHOD:
        train_count = cfg["sample_limits"]["train"] or len(train)
        if train_count % cfg["batch_size"] == 1:
            raise ValueError("Standard-backbone final training batch would have one image")
    if cfg["purpose"] == "formal":
        if (len(train), len(val), len({r["original_stem"] for r in train}),
                len({r["original_stem"] for r in val})) != (3104, 776, 776, 194):
            raise ValueError("Formal PCOD group membership/count changed")
        for part, archived in (("train", "train_split.csv"), ("validation", "val_split.csv")):
            if cfg["manifest_sha256"][part] != sha(PROJECT / "configs/splits" / archived):
                raise ValueError("Formal split differs from historical PCOD split")
    out = Path(cfg["output_dir"]).resolve()
    if not out.is_relative_to((HERE / "runs").resolve()):
        raise ValueError("Output must remain inside this experiment's runs directory")
    source_paths = [Path(__file__), HERE / "pcod_models.py",
                    *sorted((REFERENCE / "04_multiscale_ablation/code").rglob("*.py"))]
    if cfg["method"] == STANDARD_METHOD:
        standard = standard_module()
        source_paths += [Path(standard.__file__), standard.ROOT / "environment_lock.json"]
        # Vendor file hashes are verified before the standard model is built.
        installed = standard.verify_environment()
    else:
        installed = None
    backend_version = importlib.metadata.version("torch-directml") if cfg["device"] == "directml" else None
    return {"config_sha256": object_sha(cfg), "manifest_sha256": cfg["manifest_sha256"],
            "source_sha256": {str(path.relative_to(WORK)): sha(path) for path in source_paths},
            "versions": {"torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
                         "pillow": importlib.metadata.version("pillow"),
                         "python": sys.version, "torch_directml": backend_version,
                         "standard_backbone_dependencies": installed}}


def bounded_subset(dataset, limit):
    if limit is None:
        return dataset
    if type(limit) is not int or not 1 <= limit <= len(dataset):
        raise ValueError("Invalid sample limit")
    indices = np.linspace(0, len(dataset)-1, limit).astype(int).tolist()
    return Subset(dataset, indices)


def run_config(cfg, *, resume=False, stop_after_epoch=None):
    identity = validate_config(cfg)
    data = {}
    for part in ("train", "validation"):
        dataset = ExtensionPCODDataset(Path(cfg["dataset_root"]), split="train", input_mode="rgb_dop",
                                        image_size=cfg["image_size"], stems_file=Path(cfg[part + "_manifest"]),
                                        registration_aug=False)
        data[part] = bounded_subset(dataset, cfg["sample_limits"][part])
    return fit(cfg, data["train"], data["validation"], identity, resume=resume, stop_after_epoch=stop_after_epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        print(json.dumps(run_config(cfg, resume=args.resume, stop_after_epoch=args.stop_after_epoch)), flush=True)
    except Exception as error:
        print(json.dumps({"state": "failed", "type": type(error).__name__, "message": str(error)}), flush=True)
        raise


if __name__ == "__main__":
    main()
