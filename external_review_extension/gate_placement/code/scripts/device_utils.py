from __future__ import annotations

import torch


def get_device(device_name: str = "auto", directml_device: int = 1) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device_name == "directml":
        return get_directml_device(directml_device)
    if device_name != "auto":
        raise ValueError(f"Unsupported device option: {device_name}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        return get_directml_device(directml_device)
    except Exception:
        return torch.device("cpu")


def get_directml_device(index: int = 1) -> torch.device:
    try:
        import torch_directml
    except ImportError as exc:
        raise RuntimeError("torch-directml is not installed.") from exc

    count = torch_directml.device_count()
    if count <= 0:
        raise RuntimeError("No DirectML device was found.")
    if index >= count:
        index = count - 1
    return torch_directml.device(index)


def describe_device(device: torch.device) -> str:
    if str(device).startswith("privateuseone"):
        try:
            import torch_directml

            index = int(str(device).split(":")[-1])
            name = torch_directml.device_name(index).replace("\x00", "")
            return f"{device} ({name})"
        except Exception:
            return str(device)
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(0)})"
    return str(device)
