# -*- encoding: utf-8 -*-
"""
@File    : test_inpaint.py
@Time    : 2022/9/18 15:11
@Author  : livingmagic
@Email   : hongyong.li@tenclass.com
@software: Pycharm
"""


def get_ts_index(t_T, jump_len, jump_n_sample):
    jumps = {}
    for j in range(0, t_T - jump_len, jump_len):
        jumps[j] = jump_n_sample - 1

    t = t_T
    ts = []

    while t >= 1:
        t = t - 1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_len):
                t = t + 1
                ts.append(t)
    return ts


if __name__ == '__main__':
    total_ts = 50
    t_T = 10
    jump_len = 2
    jump_n_sample = 3

    ts = get_ts_index(t_T, jump_len, jump_n_sample)
    ts = [t + total_ts - t_T for t in ts]
    print(ts)
    ts.extend(list(range(total_ts - t_T))[::-1])
    print(ts)
    count = 0
    for t_last, t_cur in zip(ts[:-1], ts[1:]):
        if t_cur < t_last:
            print('reverse:', t_last, t_cur)
        else:
            print('forward:', t_last, t_cur)
        count += 1

    print(count)
