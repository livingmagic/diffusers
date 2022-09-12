# -*- encoding: utf-8 -*-
"""
@File    : stable_utils.py
@Time    : 2022/9/7 14:25
@Author  : livingmagic
@Email   : hongyong.li@tenclass.com
@software: Pycharm
"""
import random

import torch


def torch_randn(shape, seeds, device):
    batch_size = shape[0]
    one_shape = shape[1:]
    tensors = []
    generator = torch.Generator(device=device)
    if isinstance(seeds, int):
        seeds = [seeds + i for i in range(batch_size)]
    elif seeds is None:
        seeds = [random.randint(0, 2 ** 32) for _ in range(batch_size)]
    assert batch_size == len(seeds)
    for seed in seeds:
        tensors.append(torch.randn(one_shape, generator=generator.manual_seed(seed), device=device))

    return torch.stack(tensors), seeds


if __name__ == '__main__':
    print(torch_randn((3, 2), [1337169153, 4095449017, 1328036743], "cpu"))
    generator = torch.Generator(device="cpu").manual_seed(1337169153)
    print(torch.randn((2,), generator=generator, device="cpu"))
    generator = torch.Generator(device="cpu").manual_seed(4095449017)
    print(torch.randn((2,), generator=generator, device="cpu"))
    generator = torch.Generator(device="cpu").manual_seed(1328036743)
    print(torch.randn((2,), generator=generator, device="cpu"))
