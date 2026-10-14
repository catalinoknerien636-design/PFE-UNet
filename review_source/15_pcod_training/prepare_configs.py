"""Prepare PCOD checks and a disabled formal matrix; this script never trains."""
import json
from pathlib import Path

import train_pcod as t


def base_config(method, seed, purpose):
    split = t.PROJECT / "configs/splits"
    manifests = {"train": split / "train_split.csv", "validation": split / "val_split.csv"}
    run_id = f"pcod_{'hw' if purpose == 'hardware_smoke' else 'formal'}_v1_{method}_seed{seed}"
    config = {"run_id": run_id, "purpose": purpose, "method": method, "seed": seed,
        "image_size": 352, "base_channels": t.FORMAL_WIDTHS[method], "batch_size": 4,
        "epochs": 2 if purpose == "hardware_smoke" else 50,
        "learning_rate": 1e-4, "weight_decay": 1e-4, "horizontal_flip_probability": 0,
        "loss": "historical_bce_dice_v1", "sampling": "legacy_global_torch_rng",
        "device": "directml", "device_index": 1, "expected_device_name": "AMD Radeon RX 9070 XT",
        "dataset_root": str(t.PROJECT / "datasets/PCOD_1200/full-dataset"),
        **{key + "_manifest": str(path) for key, path in manifests.items()},
        "manifest_sha256": {key: t.sha(path) for key, path in manifests.items()},
        "output_dir": str(t.HERE / "runs" / run_id), "training_authorized": True,
        "protocol_frozen": purpose != "hardware_smoke",
        "sample_limits": {"train": 32 if purpose == "hardware_smoke" else None,
                          "validation": 8 if purpose == "hardware_smoke" else None}}
    if method == t.STANDARD_METHOD:
        config["model_spec"] = dict(t.standard_module().MODEL_SPEC)
    return config


def write_once(path, config):
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError(f"Existing config differs: {path}")
    t.write_json(path, config)


def main():
    smoke, formal = [], []
    source_hashes = {}
    for method in ("pfe_reference", "global_film_style", "wide_direct", t.STANDARD_METHOD):
        config = base_config(method, 42, "hardware_smoke")
        identity = t.validate_config(config)
        source_hashes.update(identity["source_sha256"])
        path = t.HERE / "configs" / (config["run_id"] + ".json")
        write_once(path, config)
        smoke.append({"path": str(path), "sha256": t.sha(path)})
    for method in ("global_film_style", "wide_direct", t.STANDARD_METHOD):
        seeds = (42, 2026, 3407) if method == t.STANDARD_METHOD else (42, 2026, 3407, 7819, 12011)
        for seed in seeds:
            config = base_config(method, seed, "formal")
            t.validate_config(config)
            config["training_authorized"] = False
            config["protocol_frozen"] = False
            path = t.HERE / "candidates" / (config["run_id"] + ".json")
            write_once(path, config)
            formal.append({"path": str(path), "sha256": t.sha(path)})
    t.write_json(t.HERE / "prepared_queues.json", {"hardware_checks": smoke,
        "formal_candidates_disabled": formal, "source_sha256": source_hashes,
        "formal_launch_allowed": False, "test_evaluation": False,
        "requirements": ["historical real-input parity", "full-resolution GPU checks", "GPU resume check",
                          "explicitly freeze reviewed formal protocol before enabling candidate configs"]})
    print(json.dumps({"hardware_configs": len(smoke), "disabled_formal_candidates": len(formal), "started": False}))


if __name__ == "__main__":
    main()
