import datetime
import time
import os
import numpy as np
import random 

def state_to_index(valuation, base): 
    """
    Maps states to an decimal integers.
    """
    factor = 1
    integer = 0
    for i in range(len(valuation)):
        integer += factor * valuation[i]
        factor *= base
    return integer

def timestamp(fmt='%y%m%dT%H%M%S'):
    """
    Returns current timestamp.
    """
    return datetime.datetime.fromtimestamp(time.time()).strftime(fmt)

def execution_to_file(data, k, x, path):
    # Create folders
    folder = os.path.join(path, str(k))
    os.makedirs(folder, exist_ok=True)
    # Create file.
    file_path = os.path.join(folder, str(x) )
    execution_file = open(file_path, 'w')
    execution_file.write('')
    execution_file.close()
    execution_file = open(file_path, 'a')
    for run_matrix in data:
        execution_file.write('#--------------------------------------------------#\n')
        for it in range(run_matrix.shape[1]):
            state = run_matrix[:,it]
            string_state = ''
            for node in state:
                string_state += str(int(node)) + ','
            string_state = string_state[:-1]
            execution_file.write(string_state + '\n')
    execution_file.close()




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