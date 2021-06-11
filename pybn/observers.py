import numpy as np
from abc import ABC, abstractmethod
from pybn.functions import plogp
from pybn.networks import FuzzyBooleanNetwork

####################################################################################################
####################################################################################################
####################################################################################################

class Observer(ABC):

    def __init__(self, nodes=1, runs=1):
        self.observations = []
        self.nodes = nodes
        self.runs = runs
        self.data = np.zeros((self.runs, self.nodes))
        self.counter = 0
        self.current_run = -1
        self.requires_update = False

    @classmethod
    @abstractmethod
    def from_configuration(cls, configuration):
        pass

    @abstractmethod
    def update(self):
        """
        Updates observer's inner state with an observed state of the network.
        """
        pass

    def post_update(self):
        self.counter += 1
        self.requires_update = True

    @abstractmethod
    def clear(self):
        """
        Clears temporary data for the next run.
        """
        pass

    def post_clear(self):
        self.current_run += 1
        self.counter = 0
        if (self.current_run == self.runs):
            self.runs += 1
            entry_shape = tuple([1])+self.data.shape[1:]
            self.data = np.append(self.data, np.zeros(entry_shape), axis=0)

    @abstractmethod
    def reset(self):
        """
        Resets the observer to its default state.
        """
        pass

    def post_reset(self):
        self.counter = 0
        self.current_run = -1
        self.data = np.zeros(self.data.shape)
        self.requires_update = False

    @abstractmethod
    def observations_to_data(self, observation_name):
        pass

    def summary(self, precision=3, nodes_per_row=5, spacing_character='±'):
        """
        Returns a human-readable string containing all the relevant data obtained by the observer.
        """

        self.post_final_update()

        summary = []
        for observation in self.observations:
            observation_data = self.observations_to_data(observation)
            observation_data_per_node = self.observations_to_data(observation, per_node=True)
            
            # Average.
            summary += ['Network ', observation.replace('_', ' '), ':\t']
            for i in range(len(observation_data)):
                if i == len(observation_data) - 1:
                    summary += [f'{observation_data[i]:.{precision}f}', '\n']
                else:
                    summary += [f'{observation_data[i]:.{precision}f}', ' ', spacing_character, ' ']
            
            # Per node.
            summary += ['\nNetwork ', observation.replace('_', ' '), ' (per node):\n']
            for j in range(self.nodes):
                for i in range(len(observation_data_per_node)):
                    if i == len(observation_data_per_node) - 1:
                        summary += [f'{observation_data_per_node[i][j]:.{precision}f}', ',\t']
                    else:
                        summary += [f'{observation_data_per_node[i][j]:.{precision}f}', ' ', spacing_character, ' ']
                if ((j+1)%nodes_per_row == 0):
                    summary += '\n'
            summary += '\n\n'
        summary = ''.join(summary)
        print(summary)

    def file_summary(self, per_node=False, precision=6, spacing_character=','):
        """
        Returns a list of pair containing all the relevant data obtained by the observer. The first element of the pair is the data name.
        """

        #self.post_final_update()

        file_summary = []
        for observation in self.observations:
            observation_summary = []
            # Per node.
            if (per_node):
                observation_data_per_node = self.observations_to_data(observation, per_node=True)
                for j in range(self.nodes):
                    for i in range(len(observation_data_per_node)):
                        observation_summary += [f'{observation_data_per_node[i][j]:.{precision}f}', spacing_character]
                observation_summary[-1] = '\n'
            # Average.
            else:
                observation_data = self.observations_to_data(observation)
                for i in range(len(observation_data)):
                    observation_summary += [f'{observation_data[i]:.{precision}f}', spacing_character]
                observation_summary[-1] +=  '\n'
            # Combine and append observation string.
            observation_summary = ''.join(observation_summary)
            file_summary.append((observation, observation_summary))
        return file_summary

    def post_final_update(self):
        if self.requires_update:
            self.process_data()
            self.requires_update = False

    @abstractmethod
    def process_data(self):
        pass

####################################################################################################
####################################################################################################
####################################################################################################

class EntropyObserver(Observer):

    def __init__(self, nodes=1, runs=1, base=2):
        super().__init__(nodes=nodes, runs=runs)
        self.observations = ['entropy', 'complexity']
        self.base = base
        self.table = np.zeros((self.nodes, self.base))

    @classmethod
    def from_configuration(cls, configuration):
        """
        Create a Entropy Observer from a configuration dictionary.
        """
        if 'base' in configuration['parameters']:
            base = configuration['parameters']['base']
        else: 
            base = 2  
        return cls(
            nodes=configuration['parameters']['nodes'], 
            runs=configuration['execution']['samples'], 
            base=base)

    def clear(self):
        self.table = np.zeros((self.nodes, self.base))
        self.post_clear()

    def reset(self):
        self.table = np.zeros((self.nodes, self.base))
        self.post_reset()

    def update(self, state):
        for i in range(len(state)):
            self.table[i,state[i]] += 1
        self.post_update()

    def observations_to_data(self, observation_name, per_node=False):
        if (observation_name == self.observations[0]):
            return self.entropy(per_node=per_node)
        elif(observation_name == self.observations[1]):
            return self.complexity(per_node=per_node)

    def process_data(self):
        probabilities = self.get_probabilities()
        run_entropy = plogp(probabilities, self.base)
        run_entropy = np.sum(run_entropy, axis=1)
        self.data[self.current_run] = run_entropy

    # Auxiliar functions.

    def get_probabilities(self):
        """
        Returns the probability table.
        """

        return self.table / self.counter

    def entropy(self, per_node=False):
        """
        Returns average and per node entropy along all runs.
        """
        data = self.data[:self.current_run + 1, :]
        if per_node:
            return np.mean(data, axis=0), np.std(data, axis=0)
        else:
            return np.mean(data), np.std(data)

    def complexity(self, per_node=False):
        """
        Returns average and per node complexity along all runs.
        """
        data = self.data[:self.current_run + 1, :]
        complexity_data = 4 * data * (1 - data)
        if per_node:
            return np.mean(complexity_data, axis=0), np.std(complexity_data, axis=0)
        else:
            return np.mean(complexity_data), np.std(complexity_data)

####################################################################################################
####################################################################################################
####################################################################################################

class TransitionsObserver(Observer):

    def __init__(self, nodes=1, runs=1, transitions=False):
        super().__init__(nodes=nodes, runs=runs)
        self.observations = ['transition_entropy', 'transition_complexity']
        self.base = 3
        self.table = np.zeros((self.nodes, self.base))
        self.transitions = transitions
        if(self.transitions):
            self.transition_table = []

    @classmethod
    def from_configuration(cls, configuration):
        """
        Create a Transitions Observer from a configuration dictionary.
        """
        return cls(
            nodes=configuration['parameters']['nodes'], 
            runs=configuration['execution']['samples'])

    def clear(self):
        self.table = np.zeros((self.nodes, self.base))
        self.previous_state = None
        if(self.transitions):
            self.transition_table = []
        self.post_clear()

    def reset(self):
        self.table = np.zeros((self.nodes, self.base))
        if(self.transitions):
            self.transition_table = []
        self.post_reset()

    def update(self, state):
        if (self.counter == 0):
            self.previous_state = state
            self.counter += 1
            return

        transition = np.zeros(self.nodes)
        transition[self.previous_state < state] = 1
        transition[self.previous_state == state] = 0
        transition[self.previous_state > state] = -1
        if(self.transitions):
            self.transition_table.append(transition)

        self.previous_state = state
        for i in range(len(state)):
            self.table[i,int(transition[i] + 1)] += 1

        self.post_update()

    def observations_to_data(self, observation_name, per_node=False):
        if (observation_name == self.observations[0]):
            return self.entropy(per_node=per_node)
        elif(observation_name == self.observations[1]):
            return self.complexity(per_node=per_node)

    def process_data(self):
        probabilities = self.get_probabilities()
        run_entropy = plogp(probabilities, self.base)
        run_entropy = np.sum(run_entropy, axis=1)
        self.data[self.current_run] = run_entropy

    # Auxiliar functions.

    def get_probabilities(self):
        """
        Returns the probability table.
        """

        return self.table / (self.counter - 1)

    def get_transitions(self):
        """
        Returns an array with the transition transitions.
        """
        if (self.transitions):
            return np.array(self.transition_table)
        else:
            raise Exception("Saving transitions was disable when observer was created. Use the flag transitions=True when creating the observer.")

    def entropy(self, per_node=False):
        """
        Returns average and per node entropy along all runs.
        """
        data = self.data[:self.current_run + 1, :]
        if per_node:
            return np.mean(data, axis=0), np.std(data, axis=0)
        else:
            return np.mean(data), np.std(data)

    def complexity(self, per_node=False):
        """
        Returns average and per node complexity along all runs.
        """
        data = self.data[:self.current_run + 1, :]
        complexity_data = 4 * data * (1 - data)
        if per_node:
            return np.mean(complexity_data, axis=0), np.std(complexity_data, axis=0)
        else:
            return np.mean(complexity_data), np.std(complexity_data)

####################################################################################################
####################################################################################################
####################################################################################################

class BinaryTransitionsObserver(Observer):

    def __init__(self, nodes=1, runs=1, transitions=False):
        super().__init__(nodes=nodes, runs=runs)
        self.observations = ['transition_entropy', 'transition_complexity']
        self.base = 2
        self.table = np.zeros((self.nodes, self.base))
        self.transitions = transitions
        if(self.transitions):
            self.transition_table = []

    @classmethod
    def from_configuration(cls, configuration):
        """
        Create a Transitions Observer from a configuration dictionary.
        """
        return cls(
            nodes=configuration['parameters']['nodes'], 
            runs=configuration['execution']['samples'])

    def clear(self):
        """
        Clears the data of the observer.
        """
        self.table = np.zeros((self.nodes, self.base))
        if(self.transitions):
            self.transition_table = []
        self.post_clear()
    
    def reset(self):
        self.table = np.zeros((self.nodes, self.base))
        if(self.transitions):
            self.transition_table = []
        self.post_reset()

    def update(self, state):
        """
        Updates transitions from an observed state.
        """

        if (self.counter == 0):
            self.previous_state = state
            self.counter += 1
            return

        transition = np.zeros(self.nodes)
        transition[self.previous_state == state] = 0
        transition[self.previous_state != state] = 1
        if(self.transitions):
            self.transition_table.append(transition)

        self.previous_state = state
        for i in range(len(state)):
            self.table[i,int(transition[i])] += 1
        
        self.post_update()

    def observations_to_data(self, observation_name, per_node=False):
        if (observation_name == self.observations[0]):
            return self.entropy(per_node=per_node)
        elif(observation_name == self.observations[1]):
            return self.complexity(per_node=per_node)

    def process_data(self):
        probabilities = self.get_probabilities()
        run_entropy = plogp(probabilities, self.base)
        run_entropy = np.sum(run_entropy, axis=1)
        self.data[self.current_run] = run_entropy

    # Auxiliar functions.

    def get_probabilities(self):
        """
        Returns the probability table.
        """

        return self.table / (self.counter - 1)

    def get_transitions(self):
        """
        Returns an array with the transition transitions.
        """
        if (self.transitions):
            return np.array(self.transition_table)
        else:
            raise Exception("Saving transitions was disable when observer was created. Use the flag transitions=True when creating the observer.")

    def entropy(self, per_node=False):
        """
        Returns average and per node entropy along all runs.
        """
        data = self.data[:self.current_run + 1, :]
        if per_node:
            return np.mean(data, axis=0), np.std(data, axis=0)
        else:
            return np.mean(data), np.std(data)

    def complexity(self, per_node=False):
        """
        Returns average and per node complexity along all runs.
        """
        data = self.data[:self.current_run + 1, :]
        complexity_data = 4 * data * (1 - data)
        if per_node:
            return np.mean(complexity_data, axis=0), np.std(complexity_data, axis=0)
        else:
            return np.mean(complexity_data), np.std(complexity_data)

####################################################################################################
####################################################################################################
####################################################################################################