import numpy as np

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

def read_file(path, std=True):
    data = np.genfromtxt(path, delimiter=',')
    if (std):
        data_mean = data[:,np.arange(0,data.shape[1],2)]
        data_std = data[:,np.arange(1,data.shape[1],2)]
        return data_mean, data_std
    else:
        data_mean = data[:,np.arange(0,data.shape[1],2)]
        return data_mean

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