import numpy as np
import random 

def valuations_to_index(valuation, base): 
    """
    Maps a valuation to an integer value.
    """
    integer = sum((base**i) * valuation[i] for i in range(len(valuation)))
    return integer


######################################################################

def index(arr=np.array,b=np.uint8): # B**N == index((B-1)*e,b=B)+1
    m = arr.shape[0]
    x=np.flip(arr, 0)
    return sum((b**i) * x[i] for i in range(m))

def apply(truth_table=np.array, state=np.array, b=np.uint8): # apply(f,x) == f[index(x)]
    idx = index(state, b)
    return truth_table[idx] 

def functions(n=np.int,ks=np.array, b=np.uint8):
    tmp = []
    for i in range(n):
        tmp.append(np.random.randint(0, b, size=b**ks[i], dtype=np.uint8) )
    return tmp

def regulators(n=np.int,ks=np.array):
    ids=[]
    for i in range(n):
        ids.append( np.random.choice(n, size=ks[i], replace=True) ) 
    return ids

def A(f=list,reg=list,x=np.array,b=np.uint8):  # aplicar la red 
    return np.array([apply(f[i], x[np.sort(reg[i])],b) for i in range(len(f))])