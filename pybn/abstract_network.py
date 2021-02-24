import numpy as np
from abc import ABC, abstractmethod
from pybn.functions import state_to_index

class AbstractNetwork(ABC):
    
    def __init__(self):
        self.nodes = None
        self.state = None
        self.bias = None
        self.base = None
        self.adjacency_matrix = None
        self.functions = None

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

    @abstractmethod
    def create_function_evaluation(self, function_input, node_id):
        pass

    @abstractmethod
    def evaluate_function(self, node_id):
        pass

    @abstractmethod
    def step(self):
        pass

