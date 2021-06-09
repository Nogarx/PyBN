import numpy as np
import json
import ray
import pybn.observers as obs
from tqdm import tqdm
from pybn.networks import AbstractNetwork
from pybn.summary import SummaryWriter

####################################################################################################
####################################################################################################
####################################################################################################

@ray.remote
def network_execution(graph, configuration, stamp, summary_writer):
    """
    Executes multiple times the boolean networks for several steps.
    Inputs: 
    network (boolean network to execute) 
    configuration (data structure that holds all the information required to successfully execute the program) 
    """
    
    # Initialize network.
    if graph is None:
        graph_function = configuration['graph']['function']
        graph = graph_function(configuration['parameters']['nodes'], configuration['parameters']['connectivity'], seed=configuration['graph']['seed'])
    network_class = configuration['network']['class']
    network = network_class.from_configuration(graph, configuration)

    # Get execution configuration parameters.
    samples = configuration['execution']['samples']
    steps = configuration['parameters']['steps'] 
    transient = configuration['parameters']['transient']

    register_observers(network, configuration)
    for _ in range(samples):
        # Set initial state.
        network.set_initial_state(observe=False)
        # Prewarm network.
        for _ in range(transient):
            network.step(observe=False)
        # Pass the last state to the observers.
            network.update_observers()
        # Execute network.
        for i in range(steps):
            network.step(observe=True)

    # Since execution is for massive experiments we export data to files instead of returning the values.
    for observer in network.observers:
        observer.pre_summary_writer()
    summary_writer.write_summary.remote(stamp, network.observers)

def run_experiment(configuration, execution_iterator, timer=False):
    """
    Create and execute multiple networks in parallel. Each network is executed runs times for steps number of steps.
    Inputs: 
    configuration (data structure that holds all the information required to successfully execute the program) 
    """

    # Get execution configuration parameters.
    network_class = configuration['network']['class']
    graph_function = configuration['graph']['function']
    networks = configuration['execution']['networks'] 

    # Check network and graph functions are valid.
    if not issubclass(network_class, AbstractNetwork):
        raise Exception("network_class is not a valid PyBN network class.")
    if not callable(graph_function):
        raise Exception("graph is not a valid PyBN graph function.")
    if len(configuration['observers']) == 0:
        raise Exception("No observer detected. Please register an observer in the configuration dictionary before continuing.")

    # Initialize Ray.
    ray.shutdown()
    ray.init()

    # Initialize summary writter.
    summary_writer = SummaryWriter.remote(configuration)

    # Iterate through all requested connectivity values.
    if timer:
        for _ in tqdm(len(execution_iterator)):
            # Update iterator and get values.
            execution_iterator.step()
            values = execution_iterator.get_variables()
            stamp = execution_iterator.get_stamp()

            # Overwrite configuration dictionary with iterator values.
            for key in values.keys():
                if (key == 'graph'):
                    continue
                configuration['parameters'][key] = values[key]
            if 'graph' in values:
                graph = values['graph']
            elif 'graph_function' in values:
                configuration['graph']['function'] = values['graph_function']
                graph = None
            else:
                graph = None

            # Run networks
            ray.get([network_execution.remote(graph, configuration, stamp, summary_writer) for _ in range(networks)])
    else:
        while(execution_iterator.step()):
            # Update iterator and get values.
            values = execution_iterator.get_variables()
            stamp = execution_iterator.get_stamp()

            # Overwrite configuration dictionary with iterator values.
            for key in values.keys():
                if (key == 'graph'):
                    continue
                configuration['parameters'][key] = values[key]
            if 'graph' in values:
                graph = values['graph']
            elif 'graph_function' in values:
                configuration['graph']['function'] = values['graph_function']
                graph = None
            else:
                graph = None

            # Run networks
            ray.get([network_execution.remote(graph, configuration, stamp, summary_writer) for _ in range(networks)])

    # Delete lock files.
    ray.get([summary_writer.remove_locks.remote()])

    # Shutdown Ray.
    ray.shutdown()

####################################################################################################
####################################################################################################
####################################################################################################

class ExecutionIterator():

    def __init__(self, precision=2):
        self.precision = precision
        self.variables = {}
        self.last_key = None
        self.first_step = True
        self.last_step = False
        self.count = -1
        self.shape_tuple = None

    def __len__(self):
        if (self.count > 0):
            return self.count
        else:
            return 0

    def clear(self):
        self.variables = {}
        self.last_key = None
        self.first_step = True
        self.last_step = False
        self.count = -1

    def reset(self):
        self.first_step = True
        self.last_step = False
        for key in self.variables.keys():
            self.variables[key][1] = 0

    def register_variable(self, name, values):
        if (not len(values) > 0):
            raise Exception("Variable values are not iterable. Values must be a non-empty list, a range or a numpy arange")
        if ('graph' and 'graph_function' in self.variables):
            raise Exception("Graph function already declared.")
        if ('graph_function' and 'graph' in self.variables):
            raise Exception("A list of graphs was already declared.")

        self.variables[name] = [values, 0]
        self.last_key = name
        if (self.count == -1):
            self.count = len(values)
        else:
            self.count *= len(values)

    def step(self):
        if (not len(self.variables) > 0):
            raise Exception("Iterator is empty.")
        if (self.first_step):
            self.first_step = False
            return True
        elif (self.last_step):
            return False

        for key in self.variables.keys():
            self.variables[key][1] += 1
            if (len(self.variables[key][0]) == self.variables[key][1]):
                self.variables[key][1] = 0
                if (key == self.last_key): 
                    self.last_step = True
                    return False
            else:
                return True

    def get_variables(self):
        if (not self.last_step):
            variables = {}
            for key in self.variables.keys():
                variables[key] = self.variables[key][0][self.variables[key][1]]        
            return variables
        else:
            return {}

    def get_stamp(self):
        if (not self.last_step):
            stamp = []
            for key in self.variables.keys():
                value = self.variables[key][0][self.variables[key][1]]
                if (isinstance(value, (int, np.integer))):
                    key_stamp = '[' + '_'.join([key,f"{value}"]) + ']'
                elif (isinstance(value, (float, np.floating))):
                    key_stamp = '[' + '_'.join([key,f"{value:.{self.precision}f}"]) + ']'
                else: 
                    key_stamp = '[' + '_'.join([key,str(self.variables[key][1])]) + ']'
                stamp.append(key_stamp)
            stamp = ''.join(stamp)
            return stamp
        else:
            return ''

    def get_stamp_list(self):
        self.reset()
        stamps_array = np.empty(self.shape(), dtype=object)
        while self.step():
            stamp = []
            position = []
            for key in self.variables.keys():
                value = self.variables[key][0][self.variables[key][1]]
                if (isinstance(value, (int, np.integer))):
                    key_stamp = '[' + '_'.join([key,f"{value}"]) + ']'
                elif (isinstance(value, (float, np.floating))):
                    key_stamp = '[' + '_'.join([key,f"{value:.{self.precision}f}"]) + ']'
                else: 
                    key_stamp = '[' + '_'.join([key,str(self.variables[key][1])]) + ']'
                stamp.append(key_stamp)
                position.append(self.variables[key][1])
            stamps_array[tuple(position)] = ''.join(stamp)
        return stamps_array

    def shape(self):
        if (self.shape_tuple is None):
            if (len(self.variables) > 0):
                shape = []
                for key in self.variables.keys():
                    shape.append(len(self.variables[key][0]))
                self.shape_tuple = tuple(shape)
                return self.shape_tuple
            else: 
                None
        else:
            return self.shape_tuple

####################################################################################################
####################################################################################################
####################################################################################################

def register_observers(network, configuration):
    observers = []
    for observer_type in configuration['observers']:
        observers.append(observer_type.from_configuration(configuration))
    network.attach_observers(observers)

def new_configuration():
    """
    Returns a new configuration data structure. 
    Configuration is not initially valid and must be filled by hand afterwards.
    """
    configuration = {
        'network': {'class': None, 'seed': None},
        'graph': {'function': None, 'seed': None},
        'fuzzy': {'conjunction': lambda x,y : min(x,y), 'disjunction': lambda x,y : max(x,y), 'negation': lambda x : 1 - x},
        'parameters': {'nodes': 0, 'basis': 0, 'bias': 0.5,'connectivity': 0, 'steps': 0, 'transient': 0},
        'summary':{'per_node': False, 'precision': 6},
        'execution': {'networks': 0, 'samples': 0},
        'observers': [],
        'storage_path' : './'
    }
    return configuration

####################################################################################################
####################################################################################################
####################################################################################################