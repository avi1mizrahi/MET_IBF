import random

import numpy as np


hash_type = np.uint64
MASK32 = hash_type(0xFFFFFFFF)

multi_shifts = np.array([
    [
        0x7F4A7C15,  # 0x9E3779B97F4A7C15,
        0x1CE4E5B9,  # 0xBF58476D1CE4E5B9,
        0x133111EB,  # 0x94D049BB133111EB,
    ],
    [
        30, 27, 31
    ]
], dtype=hash_type)


def splitmix(x, seed):
    out = hash_type(seed) & MASK32
    adder = hash_type(x) & MASK32

    for i, (mult, shift) in enumerate(multi_shifts.T):
        out *= mult
        out += adder * (i == 0)
        out ^= (out >> shift)
        out &= MASK32

    return out


def hash_sample(x, x_range, k):
    return random.Random(x).sample(x_range, k)
