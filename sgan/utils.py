import os
import time
import torch
import numpy as np
import inspect
from contextlib import contextmanager


def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def find_nan(variable, var_name):
    variable_n = variable.data.cpu().numpy()
    if np.isnan(variable_n).any():
        exit('%s has nan' % var_name)


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)


def lineno():
    return str(inspect.currentframe().f_back.f_lineno)


def resolve_device(device_preference="auto", use_gpu=True):
    """
    Decide execution device with preference order:
    1) explicit device string (cpu/cuda/mps)
    2) cuda if available
    3) mps if available
    4) cpu fallback
    """
    preferred = (device_preference or "auto").lower()
    if preferred != "auto":
        return torch.device(preferred)
    if not use_gpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch, device):
    return [tensor.to(device) for tensor in batch]


def synchronize(device=None):
    if device is None:
        return
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def get_total_norm(parameters, norm_type=2):
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            try:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm**norm_type
                total_norm = total_norm**(1. / norm_type)
            except:
                continue
    return total_norm


@contextmanager
def timeit(msg, should_time=True, device=None):
    if should_time:
        synchronize(device)
        t0 = time.time()
    yield
    if should_time:
        synchronize(device)
        t1 = time.time()
        duration = (t1 - t0) * 1000.0
        print('%s: %.2f ms' % (msg, duration))


def get_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated()
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory()
    return 0


def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir)
    return os.path.join(_dir, 'datasets', dset_name, dset_type)


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)
