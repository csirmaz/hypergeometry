

def select_of(num, max):
    """Yield all sets of num numbers from [0,max) without replacement"""
    r = list(range(num))
    while True:
        yield r
        stepped = False
        for i in range(num-1, -1, -1):
            localmax = max - num + i
            if r[i] < localmax:
                r[i] += 1
                for j in range(i+1, num, 1):
                    r[j] = r[i] - i + j
                stepped = True
                break
        if not stepped:
            break
        
