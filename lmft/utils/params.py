from typing import List
from collections import defaultdict
import re

import torch


def num_seq(nums):
    seq = []
    last_n = -99999
    for n in sorted(nums):
        if n == last_n + 1:
            seq[-1][1] = n
        else:
            seq.append([n, None])
        last_n = n
    return '[' + ','.join([str(s) if e is None else f'{s}-{e}' for s, e in seq]) + ']'


def param_names(module_or_list: torch.nn.Module | List[str], trainable=True):
    # group parameters to human-readable formats
    if isinstance(module_or_list, torch.nn.Module):
        params = [
            (name, tuple(pa.shape))
            for name, pa in module_or_list.named_parameters() if pa.requires_grad or not trainable
        ]
    else:
        params = [(n, None) for n in module_or_list]
    seen, ret, nums = set(), [], defaultdict(list)
    for name, shape in params:
        if re.findall(r'\d+', name):
            n = int(re.findall(r'\d+', name)[0])
            name = re.sub(r'(\d+)', '#', name, 1)
            nums[name].append(n)
        if name not in seen:
            seen.add(name)
            ret.append([name, shape])
    return [(name if name not in nums else name.replace('#', num_seq(nums[name])), sha) for name, sha in ret]


def param_to_buffer(module):
    """Turns all parameters of a module into buffers."""
    modules = module.modules()
    module = next(modules)
    for name, param in list(module.named_parameters(recurse=False)):
        if param.requires_grad:
            continue
        # Unregister parameter
        delattr(module, name)
        module.register_buffer(name, param)
    for module in modules:
        param_to_buffer(module)
