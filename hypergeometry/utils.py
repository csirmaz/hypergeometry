

def select_of(num, max):
    """Yield all sets of num numbers from [0,max) without replacement"""
    # WARNING The yielded list is mutated
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
        

def loop_bin(num):
    """Yield all binary variations of `num` length"""
    # WARNING The yielded list is mutated
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


def natural_bin(dim, ix):
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

        
# Unit tests
if __name__ == "__main__":
    
    def clonelist(iterator):
        return [list(x) for x in iterator]

    assert clonelist(loop_bin(1)) == [[0], [1]]
    assert clonelist(loop_bin(2)) == [[0,0], [0,1], [1,0], [1,1]]
    assert clonelist(select_of(2,4)) == [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
    assert list(loop_natural_bin(2)) == [[0,0], [0,1], [1,1], [1,0]]
    assert list(loop_natural_bin(3)) == [[0,0,0], [0,0,1], [0,1,1], [0,1,0], [1,1,0], [1,1,1], [1,0,1], [1,0,0]]
