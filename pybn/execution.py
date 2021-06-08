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
    summary_writer = SummaryWriter(configuration)

    # Iterate through all requested connectivity values.
    if timer:
        for _ in tqdm(len(execution_iterator)):
            # Update iterator and get values.
            execution_iterator.Step()
            values = execution_iterator.GetVariables()
            stamp = execution_iterator.GetStamp()

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
        while(execution_iterator.Step()):
            # Update iterator and get values.
            values = execution_iterator.GetVariables()
            stamp = execution_iterator.GetStamp()

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

    def __len__(self):
        if (self.count > 0):
            return self.count
        else:
            return 0

    def Clear(self):
        self.variables = {}
        self.last_key = None
        self.first_step = True
        self.last_step = False
        self.count = -1

    def Reset(self):
        for key in self.variables.keys():
            self.variables[key][1] = 0

    def RegisterVariable(self, name, values):
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

    def Step(self):
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

    def GetVariables(self):
        if (not self.last_step):
            variables = {}
            for key in self.variables.keys():
                variables[key] = self.variables[key][0][self.variables[key][1]]        
            return variables
        else:
            return {}

    def GetStamp(self):
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
        'execution': {'networks': 0, 'samples': 0},
        'observers': [],
        'storage_path' : './'
    }
    return configuration

def export_configuration_file(configuration, path):
    """
    Export configuration.
    """
    data = json.dumps(configuration)
    with open(path, 'w') as file:
        file.write(data)
        print("Export successful.")

def import_configuration_file(path):
    """
    Import configuration.
    """
    with open(path, "r") as file:
        data = json.load(file)
        print("Import successful.")
    return data

####################################################################################################
####################################################################################################
####################################################################################################