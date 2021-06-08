import numpy as np
from abc import ABC, abstractmethod
from pybn.functions import plogp

####################################################################################################
####################################################################################################
####################################################################################################

class Observer(ABC):

    def __init__(self):
        self.observations = None
        self.table = None
        self.counter = None
        self.current_run = None
        self.runs = None
        self.base = None
        self.nodes = None

    @classmethod
    @abstractmethod
    def from_configuration(cls, configuration):
        pass

    @abstractmethod
    def update(self, is_final_step=False):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def file_summary(self):
        pass

    @abstractmethod
    def pre_summary_writer(self):
        pass

####################################################################################################
####################################################################################################
####################################################################################################


class EntropyObserver(Observer):

    def __init__(self, nodes=1, runs=1, base=2):
        self.observations = ['entropy', 'complexity']
        self.table = np.zeros((nodes, base))
        self.counter = 0
        self.current_run = -1
        self.runs = runs
        self.base = base
        self.nodes = nodes
        self.data = np.zeros((runs, nodes))
        self.table_requires_update = False

    @classmethod
    def from_configuration(cls, configuration):
        """
        Create a Entropy Observer from a configuration dictionary.
        """
        return cls(
            nodes=configuration['parameters']['nodes'], 
            runs=configuration['execution']['samples'], 
            base=configuration['parameters']['basis'])

    def clear(self):
        """
        Clears the data of the probability table for the next run.
        """
        if (self.table_requires_update):
            self.process_table()
        self.table = np.zeros((self.nodes, self.base))
        self.current_run += 1
        self.counter = 0

    def update(self, state):
        """
        Updates probability from an observed state.
        """

        for i in range(len(state)):
            self.table[i,state[i]] += 1
        self.counter += 1
        self.table_requires_update = True

    def get_probabilities(self):
        """
        Returns the probability table.
        """

        return self.table / self.counter

    def process_table(self):
        """
        Updates data table with current run entropy.
        """

        probabilities = self.get_probabilities()
        run_entropy = plogp(probabilities, self.base)
        run_entropy = np.sum(run_entropy, axis=1)
        self.data[self.current_run] = run_entropy
        self.table_requires_update = False

    def entropy(self, std=False):
        """
        Returns average and per node entropy along all runs.
        """

        if (self.table_requires_update):
            self.process_table()

        data = self.data[:self.current_run + 1, :]
        if std is False:
            return np.mean(data), np.mean(data, axis=0)
        else:
            return (np.mean(data), np.std(data)), (np.mean(data, axis=0), np.std(data, axis=0))

    def complexity(self,  std=False):
        """
        Returns average and per node complexity along all runs.
        """

        if (self.table_requires_update):
            self.process_table()

        data = self.data[:self.current_run + 1, :]
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

        print(summary)

    def file_summary(self, per_node=False):
        """
        Returns a list of pair containing all the relevant data obtained by the observer. The first element of the pair is the data name.
        """

        entropy, nodes_entropy = self.entropy(std=True)
        complexity, nodes_complexity = self.complexity(std=True)

        if (per_node):

            # Entropy
            entropy_summary = []
            for i in range(len(nodes_entropy[0])):
                entropy_summary.append( f'{nodes_entropy[0][i]:.6f},{nodes_entropy[1][i]:.6f},')
            entropy_summary[-1] = entropy_summary[-1][:-1] + '\n'
            entropy_summary = ''.join(entropy_summary)

            # Complexity
            complexity_summary = []
            for i in range(len(nodes_complexity[0])):
                complexity_summary.append( f'{nodes_complexity[0][i]:.6f},{nodes_complexity[1][i]:.6f},')
            complexity_summary[-1] = complexity_summary[-1][:-1] + '\n'
            complexity_summary = ''.join(complexity_summary)

        else:

            # Entropy
            entropy_summary = f'{entropy[0]:.6f},{entropy[1]:.6f}\n'

            # Complexity
            complexity_summary = f'{complexity[0]:.6f},{complexity[1]:.6f}\n'

        return [('entropy', entropy_summary), ('complexity', complexity_summary)]

    def pre_summary_writer(self):
        self.process_table()
        self.table_requires_update = False

####################################################################################################
####################################################################################################
####################################################################################################

class TransitionsObserver(Observer):

    def __init__(self, nodes=1, runs=1, transitions=False):
        self.observations = ['transition_entropy', 'transition_complexity']
        self.base = 3
        self.table = np.zeros((nodes, self.base))
        self.counter = 0
        self.current_run = -1
        self.nodes = nodes
        self.data = np.zeros((runs, nodes))
        self.table_requires_update = False
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
        if (self.table_requires_update):
            self.process_table()
        self.table = np.zeros((self.nodes, self.base))
        self.current_run += 1
        self.counter = 0
        self.previous_state = None
        if(self.transitions):
            self.transition_table = []

    def update(self, state):
        """
        Updates transitions from an observed state.
        """

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
        self.counter += 1

        for i in range(len(state)):
            self.table[i,int(transition[i] + 1)] += 1
        self.table_requires_update = True

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

    def process_table(self):
        """
        Updates data table with current run entropy.
        """
        probabilities = self.get_probabilities()
        run_entropy = plogp(probabilities, self.base)
        run_entropy = np.sum(run_entropy, axis=1)
        self.data[self.current_run] = run_entropy
        self.table_requires_update = False

    def entropy(self, std=False):
        """
        Returns average and per node entropy along all runs.
        """

        if (self.table_requires_update):
            self.process_table()

        data = self.data[:self.current_run + 1, :]
        if std is False:
            return np.mean(data), np.mean(data, axis=0)
        else:
            return (np.mean(data), np.std(data)), (np.mean(data, axis=0), np.std(data, axis=0))

    def complexity(self,  std=False):
        """
        Returns average and per node complexity along all runs.
        """
        if (self.transitions):
            return np.array(self.transition_table)
        else:
            raise Exception("Saving transitions was disable when observer was created. Use the flag transitions=True when creating the observer.")

    def process_table(self):
        """
        Updates data table with current run entropy.
        """
        probabilities = self.get_probabilities()
        run_entropy = plogp(probabilities, self.base)
        run_entropy = np.sum(run_entropy, axis=1)
        self.data[self.current_run] = run_entropy
        self.table_requires_update = False

    def entropy(self, std=False):
        """
        Returns average and per node entropy along all runs.
        """

        if (self.table_requires_update):
            self.process_table()

        data = self.data[:self.current_run + 1, :]
        if std is False:
            return np.mean(data), np.mean(data, axis=0)
        else:
            return (np.mean(data), np.std(data)), (np.mean(data, axis=0), np.std(data, axis=0))

    def complexity(self,  std=False):
        """
        Returns average and per node complexity along all runs.
        """

        if (self.table_requires_update):
            self.process_table()

        data = self.data[:self.current_run + 1, :]
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
        summary = 'Transitions entropy:\t' + f"{entropy[0]:.3f}" + ' ± ' + f"{entropy[1]:.3f}" + '\n' + 'Nodes entropy:\n'
        for i in range(self.nodes):
            summary += f"{nodes_entropy[0][i]:.3f}" + ' ± ' + f"{nodes_entropy[1][i]:.3f}" + ',\t'
            if ((i+1)%5 == 0):
                summary += '\n'
        summary = summary[:-1] + '\n\n'

        # Complexity
        summary += 'Transitions complexity:\t' + f"{complexity[0]:.3f}" + ' ± ' + f"{complexity[1]:.3f}" + '\n' + 'Nodes entropy:\n'
        for i in range(self.nodes):
            summary += f"{nodes_complexity[0][i]:.3f}" + ' ± ' + f"{nodes_complexity[1][i]:.3f}" + ',\t'
            if ((i+1)%5 == 0):
                summary += '\n'
        summary = summary[:-1] + '\n'

        print(summary)

    def file_summary(self, per_node=False):
        """
        Returns a list of pair containing all the relevant data obtained by the observer. The first element of the pair is the data name.
        """

        entropy, nodes_entropy = self.entropy(std=True)
        complexity, nodes_complexity = self.complexity(std=True)

        if (per_node):

            # Entropy
            entropy_summary = []
            for i in range(len(nodes_entropy[0])):
                entropy_summary.append( f'{nodes_entropy[0][i]:.6f},{nodes_entropy[1][i]:.6f},')
            entropy_summary[-1] = entropy_summary[-1][:-1] + '\n'
            entropy_summary = ''.join(entropy_summary)

            # Complexity
            complexity_summary = []
            for i in range(len(nodes_complexity[0])):
                complexity_summary.append( f'{nodes_complexity[0][i]:.6f},{nodes_complexity[1][i]:.6f},')
            complexity_summary[-1] = complexity_summary[-1][:-1] + '\n'
            complexity_summary = ''.join(complexity_summary)

        else:

            # Entropy
            entropy_summary = f'{entropy[0]:.6f},{entropy[1]:.6f}\n'

            # Complexity
            complexity_summary = f'{complexity[0]:.6f},{complexity[1]:.6f}\n'

        return [('transition_entropy', entropy_summary), ('transition_complexity', complexity_summary)]

    def pre_summary_writer(self):
        self.process_table()
        self.table_requires_update = False

####################################################################################################
####################################################################################################
####################################################################################################


class BinaryTransitionsObserver(Observer):

    def __init__(self, nodes=1, runs=1, transitions=False):
        self.observations = ['binary_transition_entropy', 'binary_transition_complexity']
        self.base = 2
        self.table = np.zeros((nodes, self.base))
        self.counter = 0
        self.current_run = -1
        self.nodes = nodes
        self.data = np.zeros((runs, nodes))
        self.table_requires_update = False
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
        if (self.table_requires_update):
            self.process_table()
        self.table = np.zeros((self.nodes, self.base))
        self.current_run += 1
        self.counter = 0
        self.previous_state = None
        if(self.transitions):
            self.transition_table = []

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
        self.counter += 1

        for i in range(len(state)):
            self.table[i,int(transition[i])] += 1
        self.table_requires_update = True

    def get_probabilities(self):
        """
        Returns the probability table.
        """

        return self.table / (self.counter - 1)

    def transitions(self):
        """
        Returns an array with the transition transitions.
        """
        if (self.transitions):
            return np.array(self.transition_table)
        else:
            raise Exception("Saving transitions was disable when observer was created. Use the flag transitions=True when creating the observer.")

    def process_table(self):
        """
        Updates data table with current run entropy.
        """
        probabilities = self.get_probabilities()
        run_entropy = plogp(probabilities, self.base)
        run_entropy = np.sum(run_entropy, axis=1)
        self.data[self.current_run] = run_entropy
        self.table_requires_update = False

    def entropy(self, std=False):
        """
        Returns average and per node entropy along all runs.
        """

        if (self.table_requires_update):
            self.process_table()

        data = self.data[:self.current_run + 1, :]
        if std is False:
            return np.mean(data), np.mean(data, axis=0)
        else:
            return (np.mean(data), np.std(data)), (np.mean(data, axis=0), np.std(data, axis=0))

    def complexity(self,  std=False):
        """
        Returns average and per node complexity along all runs.
        """

        if (self.table_requires_update):
            self.process_table()

        data = self.data[:self.current_run + 1, :]
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
        summary = 'Transitions entropy:\t' + f"{entropy[0]:.3f}" + ' ± ' + f"{entropy[1]:.3f}" + '\n' + 'Nodes entropy:\n'
        for i in range(self.nodes):
            summary += f"{nodes_entropy[0][i]:.3f}" + ' ± ' + f"{nodes_entropy[1][i]:.3f}" + ',\t'
            if ((i+1)%5 == 0):
                summary += '\n'
        summary = summary[:-1] + '\n\n'

        # Complexity
        summary += 'Transitions complexity:\t' + f"{complexity[0]:.3f}" + ' ± ' + f"{complexity[1]:.3f}" + '\n' + 'Nodes entropy:\n'
        for i in range(self.nodes):
            summary += f"{nodes_complexity[0][i]:.3f}" + ' ± ' + f"{nodes_complexity[1][i]:.3f}" + ',\t'
            if ((i+1)%5 == 0):
                summary += '\n'
        summary = summary[:-1] + '\n'

        print(summary)

    def file_summary(self, per_node=False):
        """
        Returns a list of pair containing all the relevant data obtained by the observer. The first element of the pair is the data name.
        """

        entropy, nodes_entropy = self.entropy(std=True)
        complexity, nodes_complexity = self.complexity(std=True)

        if (per_node):

            # Entropy
            entropy_summary = []
            for i in range(len(nodes_entropy[0])):
                entropy_summary.append( f'{nodes_entropy[0][i]:.6f},{nodes_entropy[1][i]:.6f},')
            entropy_summary[-1] = entropy_summary[-1][:-1] + '\n'
            entropy_summary = ''.join(entropy_summary)

            # Complexity
            complexity_summary = []
            for i in range(len(nodes_complexity[0])):
                complexity_summary.append( f'{nodes_complexity[0][i]:.6f},{nodes_complexity[1][i]:.6f},')
            complexity_summary[-1] = complexity_summary[-1][:-1] + '\n'
            complexity_summary = ''.join(complexity_summary)

        else:

            # Entropy
            entropy_summary = f'{entropy[0]:.6f},{entropy[1]:.6f}\n'

            # Complexity
            complexity_summary = f'{complexity[0]:.6f},{complexity[1]:.6f}\n'

        return [('transition_entropy', entropy_summary), ('transition_complexity', complexity_summary)]

        # Entropy
        summary = 'Families entropy:\t' + f"{entropy[0]:.3f}" + ' ± ' + f"{entropy[1]:.3f}" + '\n' + 'Nodes entropy:\n'
        for i in range(self.nodes):
            summary += f"{nodes_entropy[0][i]:.3f}" + ' ± ' + f"{nodes_entropy[1][i]:.3f}" + ',\t'
            if ((i+1)%5 == 0):
                summary += '\n'
        summary = summary[:-1] + '\n\n'

        # Complexity
        summary += 'Families complexity:\t' + f"{complexity[0]:.3f}" + ' ± ' + f"{complexity[1]:.3f}" + '\n' + 'Nodes entropy:\n'
        for i in range(self.nodes):
            summary += f"{nodes_complexity[0][i]:.3f}" + ' ± ' + f"{nodes_complexity[1][i]:.3f}" + ',\t'
            if ((i+1)%5 == 0):
                summary += '\n'
        summary = summary[:-1] + '\n'

        print(summary)

    def file_summary(self, per_node=False):
        """
        Returns a list of pair containing all the relevant data obtained by the observer. The first element of the pair is the data name.
        """

        entropy, nodes_entropy = self.entropy(std=True)
        complexity, nodes_complexity = self.complexity(std=True)

        if (per_node):

            # Entropy
            entropy_summary = []
            for i in range(len(nodes_entropy[0])):
                entropy_summary.append( f'{nodes_entropy[0][i]:.6f},{nodes_entropy[1][i]:.6f},')
            entropy_summary[-1] = entropy_summary[-1][:-1] + '\n'
            entropy_summary = ''.join(entropy_summary)

            # Complexity
            complexity_summary = []
            for i in range(len(nodes_complexity[0])):
                complexity_summary.append( f'{nodes_complexity[0][i]:.6f},{nodes_complexity[1][i]:.6f},')
            complexity_summary[-1] = complexity_summary[-1][:-1] + '\n'
            complexity_summary = ''.join(complexity_summary)

        else:

            # Entropy
            entropy_summary = f'{entropy[0]:.6f},{entropy[1]:.6f}\n'

            # Complexity
            complexity_summary = f'{complexity[0]:.6f},{complexity[1]:.6f}\n'

        return [('family_entropy', entropy_summary), ('family_complexity', complexity_summary)]

    def pre_summary_writer(self):
        self.process_table()
        self.table_requires_update = False

####################################################################################################
####################################################################################################
####################################################################################################


class StatesObserver(Observer):

    def __init__(self, nodes=1, steps=1, runs=1, base=2):
        self.observations = ['states']
        self.table = np.zeros((steps, nodes))
        self.counter = 0
        self.current_run = -1
        self.base = base
        self.nodes = nodes
        self.steps = steps

    @classmethod
    def from_configuration(cls, configuration):
        """
        Create a Entropy Observer from a configuration dictionary.
        """
        return cls(
            nodes=configuration['parameters']['nodes'], 
            steps=configuration['parameters']['steps'], 
            runs=configuration['execution']['samples'], 
            base=configuration['parameters']['basis'])

    def clear(self):
        """
        Clears the data of the observer.
        """

        self.table = np.zeros((self.steps, self.nodes))
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

        return [('states', summary)]

    def file_summary(self):
        dummy = 1

    def pre_summary_writer(self):
        dummy = 1


####################################################################################################
####################################################################################################
####################################################################################################