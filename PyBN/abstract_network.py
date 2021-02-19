import numpy as np
from abc import ABC, abstractmethod
from PyBN import functions

class AbstractNetwork(ABC):
    
    def __init__(self, n, b, graph):
        pass

    @abstractmethod
    def create_function_valuation(self, function_input, node_id):
        pass

    @abstractmethod
    def evaluate_function(self, function_input, node_id):
        pass

    @abstractmethod
    def get_input(self, node_id):
        pass

    @abstractmethod
    def set_initial_state(self, state=None):
        pass

    @abstractmethod
    def step(self):
        pass

