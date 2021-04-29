import datetime
import time
import os
import numpy as np
import random 
from abc import ABC, abstractmethod

####################################################################################################
####################################################################################################
####################################################################################################

class Observer(ABC):

    def __init__(self):
        self.table = None
        self.counter = None
        self.current_run = None
        self.runs = None
        self.base = None
        self.nodes = None

    @abstractmethod
    def update(self, is_final_step=False):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def summary(self):
        pass

####################################################################################################
####################################################################################################
####################################################################################################

class EntropyObserver(Observer):

    def __init__(self, nodes, runs=1, base=2):
        self.table = np.zeros((nodes, base))
        self.counter = 0
        self.current_run = -1
        self.runs = runs
        self.base = base
        self.nodes = nodes
        self.data = np.zeros((runs, nodes))
        self.nodes_entropies = []
        self.nodes_complexities = []

    def clear(self):
        """
        Clears the data of the probability table for the next run.
        """
        self.table = np.zeros((self.nodes, self.base))
        self.current_run += 1
        self.counter = 0
        self.nodes_entropies = []
        self.nodes_complexities = []

    def update(self, state):
        """
        Updates probability from an observed state.
        """

        for i in range(len(state)):
            self.table[i,state[i]] += 1
        self.counter += 1
        self.entropy_requires_update = True
        self.complexity_requires_update = True

    def get_probabilities(self):
        """
        Returns the probability table.
        """

        return self.table / self.counter

    def plogp(self, p):
        if (p == 0):
            return 0
        else:
            return -p * np.log(p) / np.log(self.base)

    def run_entropy(self): 
        """
        Computes the entropy from a a probability table.
        Return average and per node entropy.
        """

        self.nodes_entropies = []
        for node_probabilities in self.get_probabilities():
            entropy = 0
            for x in node_probabilities:
                entropy += self.plogp(x)
            self.nodes_entropies.append(entropy)
        self.nodes_entropies = np.array(self.nodes_entropies)
        self.network_entropy = np.mean(self.nodes_entropies)
        self.data[self.current_run, :] = self.nodes_entropies

    def entropy(self, std=False):
        """
        Returns average and per node entropy along all runs.
        """

        data = self.data[:self.current_run, :]
        if std is False:
            return np.mean(data), np.mean(data, axis=0)
        else:
            return (np.mean(data), np.std(data)), (np.mean(data, axis=0), np.std(data, axis=0))

    def complexity(self,  std=False):
        """
        Returns average and per node complexity along all runs.
        """

        data = self.data[:self.current_run, :]
        if std is False:
            entropy = np.mean(data)
            nodes_entropy = np.mean(data, axis=0)
            complexity = 4 * entropy * (1 - entropy)
            nodes_complexity = 4 * nodes_entropy * (1 - nodes_entropy)
            return complexity, nodes_complexity
        else:
            entropy = np.mean(data)
            entropy_std = np.std(data)
            nodes_entropy = np.mean(data, axis=0)
            nodes_entropy_std = np.std(data, axis=0)
            complexity = 4 * entropy * (1 - entropy)
            #complexity_std = 4 * (entropy_std * (2 * entropy + 1))
            complexity_std = 4 * entropy_std * (1 - entropy_std)
            nodes_complexity = 4 * nodes_entropy * (1 - nodes_entropy)
            #nodes_complexity_std = 4 * (nodes_entropy_std * (2 * nodes_entropy + 1))
            nodes_complexity_std = 4 * nodes_entropy_std * (1 - nodes_entropy_std)
            return (complexity, complexity_std), (nodes_complexity, nodes_complexity_std)

    def summary(self):
        """
        Returns a human-readable string containing all the relevant data obtained by the observer.
        """

        entropy, nodes_entropy = self.entropy(std=True)
        complexity, nodes_complexity = self.complexity(std=True)

        # Entropy
        summary = 'Network entropy:\t' + f"{entropy[0]:.3f}" + ' ± ' + f"{entropy[1]:.3f}" + '\n' + 'Nodes entropy:\n'
        for i in range(self.nodes):
            summary += f"{nodes_entropy[0][i]:.3f}" + ' ± ' + f"{nodes_entropy[1][i]:.3f}" + ',\t'
            if ((i+1)%5 == 0):
                summary += '\n'
        summary = summary[:-1] + '\n\n'

        # Complexity
        summary += 'Network complexity:\t' + f"{complexity[0]:.3f}" + ' ± ' + f"{complexity[1]:.3f}" + '\n' + 'Nodes entropy:\n'
        for i in range(self.nodes):
            summary += f"{nodes_complexity[0][i]:.3f}" + ' ± ' + f"{nodes_complexity[1][i]:.3f}" + ',\t'
            if ((i+1)%5 == 0):
                summary += '\n'
        summary = summary[:-1] + '\n'

        return summary

    def file_summary(self):
        """
        Returns a string containing all the relevant data obtained by the observer.
        """

        entropy, nodes_entropy = self.entropy(std=True)
        complexity, nodes_complexity = self.complexity(std=True)

        # Entropy
        summary = f"{entropy[0]:.6f}" + '\n' + f"{entropy[1]:.6f}" + '\n'
        for val in nodes_entropy[0]:
            summary += f"{val:.6f}" + ','
        summary = summary[:-1] + '\n'
        for std in nodes_entropy[1]:
            summary += f"{std:.6f}" + ','
        summary = summary[:-1] + '\n'

        # Complexity
        summary += f"{complexity[0]:.6f}" + '\n' + f"{complexity[1]:.6f}" + '\n'
        for val in nodes_complexity[0]:
            summary += f"{val:.6f}" + ','
        summary = summary[:-1] + '\n'
        for std in nodes_complexity[1]:
            summary += f"{std:.6f}" + ','
        summary = summary[:-1] + '\n'

        return summary

####################################################################################################
####################################################################################################
####################################################################################################

class FamiliesObserver(Observer):

    def __init__(self, nodes, steps, runs=1, base=2):
        self.table = np.zeros((steps-1, nodes))
        self.counter = 0
        self.current_run = -1
        self.base = base
        self.nodes = nodes

    def clear(self):
        """
        Clears the data of the observer.
        """
        self.table = np.zeros((steps-1, nodes))
        self.counter = 0
        self.current_run += 1
        self.previous_state = None

    def update(self, state):
        """
        Updates families from an observed state.
        """

        if (self.counter == 0):
            self.previous_state = state
            return

        family = np.zeros(self.nodes)
        family[self.previous_state < state] = 1
        family[self.previous_state == state] = 0
        family[self.previous_state > state] = -1
        self.table[self.counter, :] = family

        self.previous_state = state
        self.counter += 1

    def families(self):
        """
        Returns the family table.
        """

        return self.table[self.counter:, :]

    def summary(self):
        """
        Returns a string containing all the relevant data obtained by the observer.
        """

        families = self.families()
        summary = ''
        for family in families:
            summary += str(family) + '\n'

        return summary

####################################################################################################
####################################################################################################
####################################################################################################

class StatesObserver(Observer):

    def __init__(self, nodes, steps, runs=1, base=2):
        self.table = np.zeros((steps, nodes))
        self.counter = 0
        self.current_run = -1
        self.base = base
        self.nodes = nodes

    def clear(self):
        """
        Clears the data of the observer.
        """

        self.table = np.zeros((steps, nodes))
        self.counter = 0
        self.current_run += 1
        self.previous_state = None

    def update(self, state):
        """
        Updates state table from an observed state.
        """

        self.table[self.counter, :] = state
        self.counter += 1

    def states(self):
        """
        Returns the states table.
        """

        return self.table[self.counter:, :]


    def summary(self):
        """
        Returns a string containing all the relevant data obtained by the observer.
        """

        states = self.states()
        summary = ''
        for state in states:
            summary += str(state) + '\n'

        return summary




####################################################################################################
####################################################################################################
####################################################################################################