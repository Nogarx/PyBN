import numpy as np
from abc import ABC, abstractmethod
from pybn.functions import state_to_index, get_fuzzy_lambdas

####################################################################################################
####################################################################################################
####################################################################################################

class AbstractNetwork(ABC):
    
    def __init__(self, nodes, graph, async_order=None):
        self.nodes = nodes
        self.state = np.zeros(nodes, dtype=int)
        self.bias = 0
        self.base = 2
        self.functions = []
        self.observers = []

        # Functions. Functions are created on-demand.
        self.functions = []
        for _ in range(nodes):
            self.functions.append({})

        # Graph.
        self.adjacency_matrix = np.zeros((nodes,nodes))
        for edge in graph:
            x, y = edge
            self.adjacency_matrix[y, x] = 1
        self.masks = self.compute_masks()

        # Async order.
        if async_order == None:
            self.async_order = [[i for i in range(nodes)]]
        else:
            self.async_order = async_order

    @classmethod
    @abstractmethod
    def from_configuration(cls, configuration):
        pass

    @abstractmethod
    def create_function_evaluation(self, function_input, node_id):
        """
        Add a new value to the fuction associated to node_id on input.
        """
        pass

    @abstractmethod
    def random_state(self):
        """
        Method that creates random valid states for the network.
        """
        pass    

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

    def compute_masks(self):
        masks = []
        for node_id in range(self.nodes):
            mask = self.adjacency_matrix[node_id] == 1
            masks.append(mask)
        return masks

    def compute_function_input(self, node_id):
        """
        Returns a integer representing the input to the function associated to node_id.
        """
        valuations = self.state[self.masks[node_id]]
        function_input = state_to_index(valuations, self.base)
        return function_input

    def set_initial_state(self, state=None, observe=False):
        """
        Sets the initial state of the network. If state is none, the state is set at random.
        """
        if state is not None:
            if len(state) != self.nodes:
                raise Exception("State size is different from network state size.")
            else:
                self.state = state
        else:
            self.state = self.random_state()

        # Clear and update with the initial state all observers.
        for observer in self.observers:
            observer.clear()
            if observe:
                observer.update(self.state)

    def step(self, observe=False):
        """
        Performs a time step onto the fuzzy network.
        """
        for nodes_list in self.async_order:
            # Create temporary array to store values.
            next_state = np.zeros(self.nodes, dtype=int)
            # Compute next value for each node.
            for node_id in nodes_list:
                next_state[node_id] = self.evaluate_function(node_id)
            # Replace previous network state.
            self.state = next_state
        # Update observers.
        if observe:
            self.update_observers()

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

    def update_observers(self, end_of_run=False):
        """
        Pass the current state of the network to all register observers.
        """
        for observer in self.observers:
            if end_of_run:
                observer.post_final_update()
            else:
                observer.update(self.state)

    def reset_observers(self):
        """
        Resets all the data from the observers.
        """
        for observer in self.observers:
            observer.reset()

    def observers_summary(self):
        """
        Resets all the data from the observers.
        """
        for observer in self.observers:
            observer.summary()

####################################################################################################
####################################################################################################
####################################################################################################

class BooleanNetwork(AbstractNetwork):
    
    def __init__(self, nodes, graph, bias=0.5, async_order=None):
        """
        Create a Boolean Network with n nodes. The structure of the networks is given by graph.
        Input: nodes (number of nodes), bias (how likely functions maps inputs to the state cero), graph (structure of the network)
        """
        super().__init__(nodes, graph, async_order=async_order)
        self.bias = bias

    @classmethod
    def from_configuration(cls, graph, configuration):
        """
        Create a Boolean Network from a configuration dictionary.
        """
        return cls(configuration['parameters']['nodes'], graph, configuration['parameters']['bias'])

    def create_function_evaluation(self, function_input, node_id):
        """
        Add a new value to the fuction associated to node_id on input.
        """
        random_value = 0 if np.random.rand() < self.bias else 1 
        self.functions[node_id][function_input] = random_value

    def random_state(self):
        """
        Method that creates random valid states for the network.
        """
        return np.random.randint(0, 2, self.nodes)

####################################################################################################
####################################################################################################
####################################################################################################

class FuzzyBooleanNetwork(AbstractNetwork):
    
    def __init__(self, nodes, base, graph, async_order=None):
        """
        Create a Fuzzy Boolean Network with n nodes and base b. The structure of the networks is given by graph.
        Input: n (number of nodes), b (fuzzy base), graph (structure of the network)
        """
        super().__init__(nodes, graph, async_order=async_order)
        self.base = base
        # Lambdas. Use to create the function evaluations.
        lambdas_probabilities, lambdas = get_fuzzy_lambdas(min,max,lambda x:1-x)
        self.lambdas_probabilities = lambdas_probabilities
        self.lambdas = lambdas

    @classmethod
    def from_configuration(cls, graph, configuration):
        """
        Create a Boolean Fuzzy Network from a configuration dictionary.
        """
        fuzzy_network = cls(
            configuration['parameters']['nodes'], 
            configuration['parameters']['base'], 
            graph)
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
        # We need the current state of the network to compute a valid fuzzy function. We need to scale values to [0,1] range.
        inputs = list((self.state[self.adjacency_matrix[node_id] == 1]/ (self.base-1)))
        # Since we are representing a fuzzy function of arity k with a composition of many fuzzy binary functions
        # we require n - 1 functions (We reduce the input dimension by half each layer of the computation tree). 
        # If arity is 0 just pick a value at random from {0,1}.
        size = len(inputs) - 1
        # Push an auxiliar value, since all lambda functions are binary.
        if len(inputs) == 1:
            auxiliar_value = np.random.randint(0, self.base)
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
            # We need to scale the result to the range [0, self.base].
            # If the T-norm allows for continuous representation we need to round the result to nearest element of the base.
            random_value = int(np.rint(inputs[0] * (self.base-1)))
        else:
            random_value = np.random.randint(0, self.base, dtype=int)
        self.functions[node_id][function_input] = random_value

    def random_state(self):
        """
        Method that creates random valid states for the network.
        """
        return np.random.randint(0, self.base, self.nodes)

####################################################################################################
####################################################################################################
####################################################################################################

class ProbabilisticBooleanNetwork(AbstractNetwork):
    
    def __init__(self, nodes, graph, functions_probabilities, bias=0.5, async_order=None):
        """
        Create a Probabilistic Boolean Network with n nodes. The structure of the networks is given by graph.
        Input: n (number of nodes), functions_probabilities (list of probabilities per each function per each node), 
        bias (how likely functions maps inputs to the state cero), graph (structure of the network)
        """
        super().__init__(nodes, graph, async_order=async_order)
        self.bias = bias
        # Functions. Functions are created on-demand.
        self.functions_probabilities = functions_probabilities
        self.functions = []
        for i in range(nodes):
            node_functions = []
            for _ in range(len(functions_probabilities[i])):
                node_functions.append({})
            self.functions.append(node_functions)

    @classmethod
    def from_configuration(cls, graph, configuration):
        """
        Create a Boolean Network from a configuration dictionary.
        """
        return cls(
            configuration['parameters']['nodes'], 
            graph, 
            configuration['parameters']['functions_probabilities'], 
            configuration['parameters']['bias'])

    # Override to evaluate_function to allow for multiple functions.
    def evaluate_function(self, node_id):
        """
        Evaluates the fuction associated to node_id on input. 
        If the functions was not previously defined on input, a new valuation for the input is created.
        """
        # Get a random value.
        random = np.random.rand()
        for i in range(len(self.functions_probabilities[node_id])):
            if (random <= self.functions_probabilities[node_id][i]):
                index = i 
                break
        # Compute function input.
        function_input = self.compute_function_input(node_id)
        # Evaluate function.
        if function_input not in self.functions[node_id][index]:
            self.create_function_evaluation(function_input, node_id, index)
        return self.functions[node_id][function_input]

    def create_function_evaluation(self, function_input, node_id, function_index):
        """
        Add a new value to the fuction associated to node_id on input.
        """
        random_value = 0 if np.random.rand() < self.bias else 1 
        self.functions[node_id][function_index][function_input] = random_value

    def random_state(self):
        """
        Method that creates random valid states for the network.
        """
        return np.random.randint(0, 2, self.nodes)


####################################################################################################
####################################################################################################
####################################################################################################