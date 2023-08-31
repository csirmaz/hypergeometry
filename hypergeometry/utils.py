
import numpy as np

EPSILON = 1e-6
PERMISSIVE_EPS = 2e-4
DETERMINANT_LIMIT = 1e-17
NP_TYPE = np.float_
DEBUG = False

PROFILING = {}


def profiling(label: str, obj=None):
    # print(f"PROFILING {label} {'' if obj is None else id(obj)}")
    if label in PROFILING:
        PROFILING[label] += 1
    else:
        PROFILING[label] = 1


def print_profile():
    for l in sorted(PROFILING.keys()):
        print(f"{l}: {PROFILING[l]}")


def select_of(num: int, max: int):
    """Yield all sets of num numbers from [0,max) without replacement.
    WARNING Mutates and yields the same list."""
    r = list(range(num))
    while True:
        yield r
        for i in range(num-1, -1, -1):
            localmax = max - num + i
            if r[i] < localmax:
                r[i] += 1
                for j in range(i+1, num, 1):
                    r[j] = r[i] - i + j
                break
        else:
            break
        

def loop_bin(num: int):
    """Yield all binary variations of `num` length.
    WARNING Mutates and yields the same list."""
    r = [0 for i in range(num)]
    while True:
        yield r
        for i in range(num-1, -1, -1):
            if r[i] == 0:
                r[i] = 1
                for j in range(i+1, num, 1):
                    r[j] = 0
                break
        else:
            break


def loop_many_to(num: int, max_: int, scaled=False):
    """Yield list of `num` length where each element loops from 0 to `max-1`.
    If `scaled`, scales the results so the maximum becomes 1.
    """
    r = [0 for i in range(num)]
    max_value = max_ - 1
    while True:
        if scaled:
            yield [x/(max_-1) for x in r]
        else:
            yield list(r)
        for i in range(num-1, -1, -1):
            if r[i] < max_value:
                r[i] += 1
                for j in range(i+1, num, 1):
                    r[j] = 0
                break
        else:
            break


def natural_bin(dim: int, ix: int):
    """Return the ix'th binary list of dims length for loop_natural_bin()"""
    if dim <= 0: return []
    if dim == 1: return [ix]
    flag = (ix < 2**(dim-1))
    return [0 if flag else 1] + natural_bin(dim-1, ix if flag else 2**dim-ix-1)


def loop_natural_bin(num):
    """Yield all binary variations of num length in an order where only one bit changes,
    e.g. 000, 001, 011, 010, 110, ..."""
    for i in range(2**num):
        yield natural_bin(num, i)

