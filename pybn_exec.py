#!/usr/bin/env python
import pybn.execution as execution
from pybn.networks import FuzzyBooleanNetwork
from pybn.graphs import uniform_graph
from pybn.observers import EntropyObserver, TransitionsObserver
import numpy as np
import argparse
import re

#----------------------------------------------------------------------------

def run(networks, initial_states, num_nodes, steps, transient, base_values):
    # Configure experiment.
    connectivity_values = np.arange(1.0,5.01,0.1)
    observers = [EntropyObserver, TransitionsObserver]
    storage_path = '/storage/gershenson_g/mendez/pybn/'

    configuration = {
            'network': {'class': FuzzyBooleanNetwork},
            'graph': {'function': uniform_graph},
            'fuzzy': {'conjunction': lambda x,y : min(x,y), 'disjunction': lambda x,y : max(x,y), 'negation': lambda x : 1 - x},
            'parameters': {'nodes': num_nodes, 'steps': steps, 'transient': transient},
            'summary':{'per_node': True, 'precision': 6},
            'execution': {'networks': networks, 'samples': initial_states},
            'observers': observers,
            'storage_path' : storage_path
        }

    # Initialize iterator.
    iterator = execution.ExecutionIterator()
    iterator.register_variable('base', base_values)
    iterator.register_variable('k', connectivity_values)

    # Dispatch the experiment.
    execution.run_experiment(configuration, iterator)

#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

_cmdline_help_epilog = '''Example:
  # Example of experiment.
  python %(prog)s ---networks=1000 --initial_states=1000 --num_nodes=40 --steps=500 --transient=250 --base_values=2,3,4
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run a PyBN experiment.',
        epilog=_cmdline_help_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    group = parser.add_argument_group('General configuration')
    group.add_argument('--networks', dest='networks', type=int, help='Number of networks per configuration tuple', default=1000)
    group.add_argument('--initial_states', dest='initial_states', type=int, help='Number of initial states per network', default=1000)
    group.add_argument('--num_nodes', dest='num_nodes', type=int, help='Number of nodes per network', default=40)
    group.add_argument('--steps', dest='steps', type=int, help='Number of steps per network run', default=500)
    group.add_argument('--transient', dest='transient', type=int, help='Number of pre-warm steps per network run', default=250)
    group.add_argument('--base_values', dest='base_values', type=_parse_num_range, help='List of bases for the experiment', default=[2])

    args = parser.parse_args()
    try:
        run(**vars(args))
    except:
        print(f'Error: Arguments mismatch.')
        exit(1)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
