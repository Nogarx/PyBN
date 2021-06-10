import numpy as np
from math import ceil
from operator import itemgetter 

####################################################################################################
####################################################################################################
####################################################################################################

def uniform_graph(nodes, k, seed=None, **kwargs):
    """
    Input: nodes (nodes), k (average connectivity).
    Output: list of adjacency representing a graph with nodes nodes and average connectivity k.
    """

    # Seed for reproducibility.
    if seed is not None:
        np.random.seed(seed)

    # Required number of edges to get average connectivity.
    number_edges = ceil(nodes * k)

    # Create tuples with all possible edges. Select number_edges from edges.
    edges = []
    for i in range(nodes):
        for j in range(nodes):
            edges.append((i,j))
    indices = np.arange(nodes*nodes)
    edges_list = np.random.choice(indices, size=number_edges, replace=False)

    # Get edges and sort them.
    if len(edges_list) == 1:
        adjacency_list = itemgetter(*edges_list)(edges)
        adjacency_list = [adjacency_list]
    elif len(edges_list) > 1:
        adjacency_list = list(itemgetter(*edges_list)(edges))
        adjacency_list = sorted(adjacency_list)
    else:
        adjacency_list = []
        
    return adjacency_list

def regular_graph(nodes, k, seed=None, **kwargs):
    """
    Input: nodes (nodes), k (node connectivity).
    Output: list of adjacency representing a regular graph with nodes nodes and k edges.
    """

    # Seed for reproducibility.
    if seed is not None:
        np.random.seed(seed)

    # Create tuples with all possible edges. Select number_edges from edges.
    adjacency_list = []
    indices = np.arange(nodes)
    number_edges = k
    for i in range(nodes):
        edges = []
        for j in range(nodes):
            edges.append((i,j))
        edges_list = np.random.choice(indices, size=number_edges, replace=False)
        if len(edges_list) == 1:
            node_adjacency_list = itemgetter(*edges_list)(edges)
            adjacency_list += [node_adjacency_list]
        elif len(edges_list) > 1:
            node_adjacency_list = sorted(list(itemgetter(*edges_list)(edges)))
            adjacency_list += node_adjacency_list
        
    return adjacency_list

def power_law_graph(nodes, k, w, seed=None, **kwargs):
    """
    Input: nodes (nodes), k (node maximum connectivity), w (decay rate).
    Output: list of adjacency representing a power law graph with nodes nodes computed where the amount of edges for each node 
    is computed from the function ceil(k*(i+1)^(-w/nodes)), where i is the index of the node.
    """

    # Seed for reproducibility.
    if seed is not None:
        np.random.seed(seed)

    # Create tuples with all possible edges. Select number_edges from edges.
    adjacency_list = []
    indices = np.arange(nodes)
    for i in range(nodes):
        edges = []
        for j in range(nodes):
            edges.append((i,j))
        number_edges = ceil(k * (i+1) ** (-w/nodes) )
        edges_list = np.random.choice(indices, size=number_edges, replace=False)
        if len(edges_list) == 1:
            node_adjacency_list = itemgetter(*edges_list)(edges)
            adjacency_list += [node_adjacency_list]
        elif len(edges_list) > 1:
            node_adjacency_list = sorted(list(itemgetter(*edges_list)(edges)))
            adjacency_list += node_adjacency_list

    return adjacency_list

####################################################################################################
####################################################################################################
####################################################################################################