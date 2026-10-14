"""Explicit half-pixel bilinear resize candidate, not approved for training."""
import types
import torch
from torch.nn import functional as F


def axis_weights(source, target, axis, tensor):
    coordinate = (torch.arange(target, dtype=torch.float64) + .5) * (source / target) - .5
    coordinate = coordinate.clamp(0, source - 1)
    low = coordinate.floor().long()
    high = (low + 1).clamp_max(source - 1)
    shape = [1, 1, 1, 1]
    shape[axis] = target
    weight = (coordinate - low).to(dtype=tensor.dtype).reshape(shape).to(tensor.device)
    return low.to(tensor.device), high.to(tensor.device), weight


def bilinear_half_pixel(tensor, size):
    if tensor.ndim != 4 or not tensor.is_floating_point():
        raise ValueError("Expected floating NCHW input")
    if len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
        raise ValueError("Expected positive integer output height/width")
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    result = tensor
    for axis, target in ((3, size[1]), (2, size[0])):
        source = result.shape[axis]
        if source == target:
            continue
        low, high, weight = axis_weights(source, target, axis, result)
        a, b = result.index_select(axis, low), result.index_select(axis, high)
        result = a * (1 - weight) + b * weight
    return result


def portable_up_forward(self, x, skip):
    x = bilinear_half_pixel(x, tuple(skip.shape[-2:]))
    return self.conv(torch.cat([skip, x], dim=1))


def install_unet_candidate(model):
    count = 0
    for module in model.modules():
        if type(module).__name__ == "Up" and type(module).__module__ == "models.unet":
            module.forward = types.MethodType(portable_up_forward, module)
            count += 1
    if count != 4:
        raise ValueError(f"Expected four reference U-Net upsampling modules, got {count}")
    return model
