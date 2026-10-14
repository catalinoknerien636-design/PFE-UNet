"""Half-pixel 2x resize using repeated samples and a separable fixed filter."""
import torch
from resize_candidate import bilinear_half_pixel as generic_resize


def bilinear_half_pixel(tensor, size):
    if tuple(size) != (tensor.shape[-2] * 2, tensor.shape[-1] * 2):
        return generic_resize(tensor, size)
    result = tensor
    for axis in (3, 2):
        result = result.repeat_interleave(2, dim=axis)
        first = result.narrow(axis, 0, 1)
        last = result.narrow(axis, result.shape[axis] - 1, 1)
        left = torch.cat((first, result.narrow(axis, 0, result.shape[axis] - 1)), dim=axis)
        right = torch.cat((result.narrow(axis, 1, result.shape[axis] - 1), last), dim=axis)
        result = .25 * left + .5 * result + .25 * right
    return result
