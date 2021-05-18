import numpy as np
import random 

####################################################################################################
####################################################################################################
####################################################################################################

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

def plogp(p, base):
    p = np.ma.array(p, mask=(p<=0))
    return -p * np.log(p, where=(p.mask==False)) / np.log(base)

####################################################################################################
####################################################################################################
####################################################################################################

def get_fuzzy_lambdas(conjunction, disjunction, negation):
    fuzzy_lambdas_probabilities = [17/84,4/84,4/84,4/84,4/84,4/84,4/84,4/84,4/84,4/84,4/84,4/84,4/84,1/84,1/84,17/84]
    fuzzy_lambdas = [
        lambda x, y : 1,
        lambda x, y : disjunction(x,y),
        lambda x, y : disjunction(x,negation(y)),
        lambda x, y : disjunction(negation(x),y),
        lambda x, y : disjunction(negation(x),negation(y)),
        lambda x, y : conjunction(x,y),
        lambda x, y : conjunction(x,negation(y)),
        lambda x, y : conjunction(negation(x),y),
        lambda x, y : conjunction(negation(x),negation(y)),
        lambda x, y : x,
        lambda x, y : negation(x),
        lambda x, y : y,
        lambda x, y : negation(y),
        lambda x, y : disjunction(conjunction(x,y),conjunction(negation(x),negation(y))),
        lambda x, y : conjunction(conjunction(x,negation(y)),conjunction(negation(x),y)),
        lambda x, y : 0]
    return fuzzy_lambdas_probabilities, fuzzy_lambdas

####################################################################################################
####################################################################################################
####################################################################################################

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
