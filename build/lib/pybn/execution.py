import numpy as np
import json
import ray
import pybn.observers as obs
from tqdm import tqdm
from pybn.networks import AbstractNetwork
from pybn.summary import SummaryWriter

from functools import partial
from joblib import Parallel, delayed

####################################################################################################
####################################################################################################
####################################################################################################

@ray.remote
def network_execution(connectivity, graph, configuration, summary_writer):
    """
    Executes multiple times the boolean networks for several steps.
    Inputs: 
    network (boolean network to execute) 
    configuration (data structure that holds all the information required to successfully execute the program) 
    """
    
    # Initialize network.
    if graph is None:
        graph_function = configuration['graph']['function']
        graph = graph_function(configuration['network']['nodes'], connectivity, seed=configuration['graph']['seed'])
    network_class = configuration['network']['class']
    network = network_class.from_configuration(graph, configuration)

    # Get execution configuration parameters.
    runs = configuration['execution']['runs']
    steps = configuration['execution']['steps'] 
    transient = configuration['execution']['transient']

    register_observers(network, configuration)
    for _ in range(runs):
        # Set initial state.
        network.set_initial_state()
        # Prewarm network.
        for _ in range(transient):
            network.step(observe=False)
        # Execute network.
        for i in range(steps):
            network.step(observe=True)

    # Since execution is for massive experiments we export data to files instead of returning the values.
    key = f"{connectivity:.2f}"
    summary_writer.write_summary.remote(key, network.observers)

def run_experiment(configuration):
    """
    Create and execute multiple networks in parallel. Each network is executed runs times for steps number of steps.
    Inputs: 
    configuration (data structure that holds all the information required to successfully execute the program) 
    """

    # Get execution configuration parameters.
    network_class = configuration['network']['class']
    nodes = configuration['network']['nodes']
    graph_function = configuration['graph']['function']
    graph_seed = configuration['graph']['seed']
    k_start = configuration['graph']['k_start']
    k_end = configuration['graph']['k_end'] 
    k_step = configuration['graph']['k_step']
    repetitions = configuration['execution']['networks'] 
    jobs = configuration['execution']['jobs'] 

    # Check network and graph functions are valid.
    if not issubclass(network_class, AbstractNetwork):
        raise Exception("network_class is not a valid PyBN network class.")
    if not callable(graph_function):
        raise Exception("graph is not a valid PyBN graph function.")

    # Initialize Ray.
    ray.shutdown()
    ray.init()

    # Initialize summary writer.
    summary_writer = SummaryWriter.remote(configuration)
    summary_writer.initialize_directory.remote()

    # Iterate through all requested connectivity values.
    for k in tqdm(np.arange(k_start, k_end + 0.5 * k_step, k_step)):
        graph = graph_function(nodes, k, seed=graph_seed) if (graph_seed is not None) else None
        ray.wait([network_execution.remote(k, graph, configuration, summary_writer) for _ in range(repetitions)])

    # Shutdown Ray.
    ray.shutdown()

####################################################################################################
####################################################################################################
####################################################################################################

def register_observers(network, configuration):
    observers = []
    if configuration['observers']['entropy']:
        observers.append(obs.EntropyObserver(
                                    configuration['network']['nodes'], 
                                    configuration['execution']['runs'], 
                                    configuration['network']['basis']))
    if configuration['observers']['family']:
        observers.append(obs.FamiliesObserver(
                                    configuration['network']['nodes'], 
                                    configuration['execution']['steps'], 
                                    configuration['execution']['runs'], 
                                    configuration['network']['basis']))
    if configuration['observers']['family']:
        observers.append(obs.StatesObserver(
                                    configuration['network']['nodes'], 
                                    configuration['execution']['steps'], 
                                    configuration['execution']['runs'], 
                                    configuration['network']['basis']))

    if len(observers) == 0:
        raise Exception("No observer detected. Please register an observer in the configuration dictionary before continuing.")

    network.attach_observers(observers)

def new_configuration():
    """
    Returns a new configuration data structure. 
    Configuration is not initially valid and must be filled by hand afterwards.
    """
    configuration = {
        'network': {'class': None, 'nodes': 0, 'basis': 0, 'bias': 0.5, 'seed': None},
        'fuzzy': {'conjunction': lambda x,y : min(x,y), 'disjunction': lambda x,y : max(x,y), 'negation': lambda x : 1 - x},
        'graph': {'function': None, 'k_start': 0, 'k_end': 0, 'k_step': 0, 'seed': None},
        'observers': {'entropy': 0, 'family': 0, 'states': 0},
        'execution': {'networks': 0, 'runs': 0, 'steps': 0, 'transient': 0, 'jobs': 1},
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