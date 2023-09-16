
import numpy as np
import random

EPSILON = 1e-7
PERMISSIVE_EPS = 1e-7
BOUNDINGBOX_EPS = 1e-7
DETERMINANT_LIMIT = 1e-15
NP_TYPE = np.float_
DEBUG = False  # Whether to print debug information about calculations. Can also be turned on/off in a nested way using debug_push/debug_pop
XCHECK = False  # Whether to do extra calculations do cross-check results
THROTTLE_LIMITS = {  # Settings for the types of throttles, see throttle()
    'xcheck_bbox': 100
}

_DEBUG_LOCK = 0
_PROFILING = {}
_THROTTLE_COUNTER = {k: random.randrange(0, v) for k, v in THROTTLE_LIMITS.items()}


def throttle(label):
    """Return true every Nth call for the given label"""
    _THROTTLE_COUNTER[label] += 1
    if _THROTTLE_COUNTER[label] >= THROTTLE_LIMITS[label]:
        _THROTTLE_COUNTER[label] = 0
        return True
    return False


def debug_push():
    """Enable debugging (and count how many times it has been enabled)"""
    global DEBUG, _DEBUG_LOCK
    _DEBUG_LOCK += 1
    print(f"Debugging entering {_DEBUG_LOCK}")
    DEBUG = True


def debug_pop():
    """Undo one debug_push"""
    global DEBUG, _DEBUG_LOCK
    _DEBUG_LOCK -= 1
    print(f"Debugging exiting {_DEBUG_LOCK}")
    if _DEBUG_LOCK <= 0:
        DEBUG = False


def profiling(label: str, obj=None):
    """Collect how many times this function is called with each label"""
    # print(f"_PROFILING {label} {'' if obj is None else id(obj)}")
    if label in _PROFILING:
        _PROFILING[label] += 1
    else:
        _PROFILING[label] = 1


def print_profile():
    print(f"PROFILING DATA")
    for l in sorted(_PROFILING.keys()):
        print(f"{l}: {_PROFILING[l]}")


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


def loop_many_to_v(max_: list[int], scaled=False):
    """Yield list of `num` length where element i loops from 0 to `max_[i]-1`.
    If `scaled`, scales the results so the maximum becomes 1-1/max_.
    """
    num = len(max_)
    r = [0 for i in range(num)]
    max_values = [x - 1 for x in max_]
    while True:
        if scaled:
            yield [(0 if x == 0 else x / max_[i]) for i, x in enumerate(r)]
        else:
            yield list(r)  # clone
        for i in range(num-1, -1, -1):
            if r[i] < max_values[i]:
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


class NotIndependentError(Exception):
    pass


class XCheckError(Exception):
    pass
