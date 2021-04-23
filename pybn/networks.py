import numpy as np
from abc import ABC, abstractmethod
from pybn.functions import state_to_index, get_fuzzy_lambdas

####################################################################################################
####################################################################################################
####################################################################################################

class AbstractNetwork(ABC):
    
    def __init__(self):
        self.nodes = None
        self.state = None
        self.bias = None
        self.base = None
        self.adjacency_matrix = None
        self.functions = None
        self.observers = []

    @classmethod
    @abstractmethod
    def from_configuration(cls, configuration):
        pass

    def compute_function_input(self, node_id):
        """
        Returns a integer representing the input to the function associated to node_id.
        """
        valuations = self.state[self.adjacency_matrix[node_id] == 1]
        function_input = state_to_index(valuations, self.base)
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
            self.state = np.random.randint(0, 2, self.nodes)

        # Clear and update with the initial state all observers.
        for observer in self.observers:
            observer.clear()
            observer.update(self.state)

    @abstractmethod
    def create_function_evaluation(self, function_input, node_id):
        """
        Add a new value to the fuction associated to node_id on input.
        """
        pass

    @abstractmethod
    def evaluate_function(self, node_id):
        """
        Evaluates the fuction associated to node_id on input. 
        If the functions was not previously defined on input, a new valuation for the input is created.
        """
        pass

    @abstractmethod
    def step(self, observe=False):
        """
        Performs a time step onto the network.
        """
        pass

    def detach_observers(self):
        """
        Detach all observers from the network.
        """
        self.observers = []

    def attach_observers(self, observers):
        """
        Detach all observers from the network.
        """
        for observer in observers:
            self.observers.append(observer)

    def update_observers(self):
        """
        Pass the current state of the network to all register observers.
        """
        for observer in self.observers:
            observer.update(self.state)

####################################################################################################
####################################################################################################
####################################################################################################

class BooleanNetwork(AbstractNetwork):
    
    def __init__(self, n, graph, bias=0.5):
        """
        Create a Boolean Network with n nodes. The structure of the networks is given by graph.
        Input: n (number of nodes), bias (how likely functions maps inputs to the state cero), graph (structure of the network)
        """
        super().__init__()
        self.nodes = n
        self.bias = bias
        self.base = 2
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
        """
        Create a Boolean Network from a configuration dictionary.
        """
        return cls(configuration['network']['nodes'], graph, configuration['network']['bias'])

    def create_function_evaluation(self, function_input, node_id):
        """
        Add a new value to the fuction associated to node_id on input.
        """
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

    def step(self, observe=False):
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
        # Update observers.
        if observe:
            self.update_observers()

####################################################################################################
####################################################################################################
####################################################################################################

class FuzzyBooleanNetwork(AbstractNetwork):
    
    def __init__(self, n, b, graph, bias=0.5):
        """
        Create a Fuzzy Boolean Network with n nodes and base b. The structure of the networks is given by graph.
        Input: n (number of nodes), b (fuzzy basis), graph (structure of the network)
        """
        super().__init__()
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
        """
        Create a Boolean Fuzzy Network from a configuration dictionary.
        """
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

    def step(self, observe=False):
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
        # Update observers.
        if observe:
            self.update_observers()

####################################################################################################
####################################################################################################
####################################################################################################