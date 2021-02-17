import numpy as np
from math import ceil
from operator import itemgetter 

def create_uniform_graph(n, k):
    """
    Input: n (nodes), k (average connectivity).
    Output: list of adjacency representing a graph with n nodes and average connectivity k.
    """

    # Required number of edges to get average connectivity.
    number_edges = ceil(n * k)

    # Create tuples with all possible edges. Select number_edges from edges.
    edges = []
    for i in range(n):
        for j in range(n):
            edges.append((i,j))
    indices = np.arange(n*n)
    edges_list = np.random.choice(indices, size=number_edges, replace=False)

    # Get edges and sort them.
    adjacency_list = list(itemgetter(*edges_list)(edges))
    adjacency_list = sorted(adjacency_list)
    return adjacency_list