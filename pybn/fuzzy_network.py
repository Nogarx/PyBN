import numpy as np
from pybn.abstract_network import AbstractNetwork
from pybn.functions import state_to_index, get_fuzzy_lambdas

####################################################################################################
####################################################################################################
####################################################################################################

class FuzzyBooleanNetwork(AbstractNetwork):
    
    def __init__(self, n, b, graph, bias=0.5):
        """
        Create a Fuzzy Boolean Network with n nodes and base b. The structure of the networks is given by graph.
        Input: n (number of nodes), b (fuzzy basis), graph (structure of the network)
        """
        self.nodes = n
        self.bias = bias
        self.base = b
        # Nodes.
        self.state = np.zeros(n, dtype=int)
        # Graph.
        self.adjacency_matrix = np.zeros((n,n))
        for edge in graph:
            x, y = edge
            self.adjacency_matrix[x, y] = 1
        # Lambdas. Use to create the function evaluations.
        lambdas_probabilities, lambdas = get_fuzzy_lambdas(min,max,lambda x:1-x)
        self.lambdas_probabilities = lambdas_probabilities
        self.lambdas = lambdas
        # Functions. Functions are created on-demand.
        self.functions = []
        for _ in range(n):
            self.functions.append({})

    @classmethod
    def from_configuration(cls, graph, configuration):
        fuzzy_network = cls(
            configuration['network']['nodes'], 
            configuration['network']['basis'], 
            graph,
            configuration['network']['bias'])
        # Replace lambdas.
        conjunction = configuration['fuzzy']['conjunction']
        disjunction = configuration['fuzzy']['disjunction']
        negation = configuration['fuzzy']['negation']
        lambdas_probabilities, lambdas = get_fuzzy_lambdas(conjunction, disjunction, negation)
        fuzzy_network.lambdas_probabilities = lambdas_probabilities
        fuzzy_network.lambdas = lambdas
        return fuzzy_network

    def create_function_evaluation(self, function_input, node_id):
        """
        Add a new value to the fuction associated to node_id on input.
        """
        # We need the current state of the network to compute a valid fuzzy function.
        inputs = list(self.state[self.adjacency_matrix[node_id] == 1])
        # Since we are representing a fuzzy function of arity k with a composition of many fuzzy binary functions
        # we require 2n - 1 functions (It a complete binary tree). If arity is 0 just pick a value at random from {0,1}.
        size = len(inputs) - 1
        # Push an auxiliar value, since all lambda functions are binary.
        if len(inputs) == 1:
            auxiliar_value = 0 if np.random.rand() < self.bias else 1
            inputs.append(auxiliar_value)
        if size > 0:
            fuzzy_functions = np.random.choice(self.lambdas, size=size, replace=True, p=self.lambdas_probabilities)
            # Iterate through all functions, evaluating back and forth through all inputs.
            index = 0
            for function in fuzzy_functions:
                inputs[index] = function(inputs[index], inputs[index + 1])
                inputs.pop(index + 1)
                index += 1
                if (index + 1 >= len(inputs)):
                    index = 0
                    inputs.reverse()
            random_value = inputs[0]
        else:
            random_value = 0 if np.random.rand() < self.bias else 1 
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

####################################################################################################
####################################################################################################
####################################################################################################