import numpy as np
from PyBN import functions

class FuzzyBooleanNetwork():
    
    def __init__(self, n, b, graph):
        """
        Create a Fuzzy Boolean Network with n nodes and base b. The structure of the networks is given by graph.
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

    def create_function_valuation(self, function_input, node_id):
        """
        Add a new value to the fuction associated to node_id on input.
        """
        random_value = np.random.randint(0, self.base)
        self.functions[node_id][function_input] = random_value

    def evaluate_function(self, function_input, node_id):
        """
        Evaluates the fuction associated to node_id on input. 
        If the functions was not previously defined on input, a new valuation for the input is created.
        """
        if function_input in self.functions[node_id]:
            # Function is defined on input.
            return self.functions[node_id][function_input]
        else:
            # Function is not defined on input.
            self.create_function_valuation(function_input, node_id)
            return self.functions[node_id][function_input]

    def get_input(self, node_id):
        """
        Returns a integer representing the input to the function associated to node_id.
        """
        valuations = self.state[self.adjacency_matrix[node_id] == 1]
        function_input = functions.valuations_to_index(valuations, self.base)
        return function_input

    def set_initial_state(self, state=None):
        """
        Sets the initial state of the network. If state is none, the state is set at random.
        """
        if state is not None:
            if len(state) != self.nodes:
                raise Exception("State size is different from network state size.")
            else:
                self.state = state
        else:
            self.state = np.random.randint(0, self.base, self.nodes)

    def step(self):
        """
        Performs a time step onto the fuzzy network.
        """
        # Create temporary array to store values.
        next_state = np.zeros(self.nodes, dtype=int)
        # Compute next value for each node.
        for node_id in range(self.nodes):
            function_input = self.get_input(node_id)
            next_state[node_id] = self.evaluate_function(function_input, node_id)
        # Replace previous network state.
        self.state = next_state

