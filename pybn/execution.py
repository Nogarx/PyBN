import numpy as np
import json
from joblib import Parallel, delayed
from pybn.functions import execution_to_file
from pybn.abstract_network import AbstractNetwork

####################################################################################################
####################################################################################################
####################################################################################################

def network_execution(k, configuration, x, graph=None):
    """
    Executes multiple times the boolean networks for several steps.
    Inputs: 
    network (boolean network to execute) 
    configuration (data structure that holds all the information required to successfully execute the program) 
    """

    # Initialize network.
    if graph is None:
        graph_function = configuration['graph']['function']
        graph = graph_function(configuration['network']['nodes'], k, seed=configuration['graph']['seed'])
    network_class = configuration['network']['class']
    network = network_class.from_configuration(graph, configuration)

    # Get execution configuration parameters.
    runs = configuration['execution']['runs']
    steps = configuration['execution']['steps'] 
    transient = configuration['execution']['transient']

    execution_data = []
    for _ in range(runs):
        # Set initial state.
        network.set_initial_state()
        data = np.zeros((network.nodes,steps))
        # Prewarm network.
        for _ in range(transient):
            network.step()
        # Execute network.
        for i in range(steps):
            network.step()
            data[:,i] = np.copy(network.state) 
        # Save run data.
        execution_data.append(data)

    # Since execution is for massive experiments we export data to files instead of returning the values.
    execution_to_file(execution_data, k, x, configuration['storage_path'])


def parallel_execution(configuration):
    """
    Create and execute multiple networks in parallel. Each network is executed runs times for steps number of steps.
    Inputs: 
    configuration (data structure that holds all the information required to successfully execute the program) 
    """

    #if __name__ == '__main__':

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

    # Iterate through all requested connectivity values.
    for k in np.arange(k_start, k_end, k_step):
        graph = graph_function(nodes, k, seed=graph_seed) if (graph_seed is not None) else None
        Parallel(n_jobs=jobs)( delayed(network_execution)(k, configuration, x, graph) for x in range(repetitions) ) 

####################################################################################################
####################################################################################################
####################################################################################################

def new_configuration():
    """
    Returns a new configuration data structure. 
    Configuration is not initially valid and must be filled by hand afterwards.
    """
    configuration = {
        'network': {'class': None, 'nodes': 0, 'basis': 0, 'bias': 0.5, 'seed': None},
        'graph': {'function': None, 'k_start': 0, 'k_end': 0, 'k_step': 0, 'seed': None},
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