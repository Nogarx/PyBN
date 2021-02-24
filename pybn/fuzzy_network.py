import numpy as np
from pybn.abstract_network import AbstractNetwork
from pybn.functions import state_to_index

class FuzzyBooleanNetwork(AbstractNetwork):
    
    def __init__(self, n, b, graph):
        """
        Create a Fuzzy Boolean Network with n nodes and base b. The structure of the networks is given by graph.
        Input: n (number of nodes), b (fuzzy basis), graph (structure of the network)
        """
        self.nodes = n
        self.base = b
        # Nodes.
        self.state = np.zeros(n, dtype=int)
        # Graph.
        self.adjacency_matrix = np.zeros((n,n))
        for edge in graph:
            x, y = edge
            self.adjacency_matrix[x, y] = 1
        # Functions. Functions are created on-demand.
        self.functions = []
        for _ in range(n):
            self.functions.append({})

    @classmethod
    def from_configuration(cls, graph, configuration):
        return cls(configuration['network']['nodes'], configuration['network']['basis'], graph)

    def create_function_evaluation(self, function_input, node_id):
        """
        Add a new value to the fuction associated to node_id on input.
        """
        random_value = np.random.randint(0, self.base)
        self.functions[node_id][function_input] = random_value

    def evaluate_function(self, node_id):
        """
        Evaluates the fuction associated to node_id on input. 
        If the functions was not previously defined on input, a new valuation for the input is created.
        """
        # Compute function input.
        function_input = self.compute_function_input(node_id)
        # Evaluate function.
        if function_input not in self.functions[node_id]:
            self.create_function_evaluation(function_input, node_id)
        return self.functions[node_id][function_input]

    def step(self):
        """
        Performs a time step onto the fuzzy network.
        """
        # Create temporary array to store values.
        next_state = np.zeros(self.nodes, dtype=int)
        # Compute next value for each node.
        for node_id in range(self.nodes):
            next_state[node_id] = self.evaluate_function(node_id)
        # Replace previous network state.
        self.state = next_state
