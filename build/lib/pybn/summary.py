import datetime
import time
import numpy as np
import random 
import ray
import os
from filelock import FileLock

####################################################################################################
####################################################################################################
####################################################################################################

@ray.remote
class SummaryWriter():

    def __init__(self, configuration):
        self.storage_path = configuration['storage_path']
        self.directory_path = None
        self.file_key = None
        self.initialize_directory()

    def initialize_directory(self):
        """
        Create a new directory to store the data of the current experiment.
        """
        self.directory_path = os.path.join(self.storage_path, self.timestamp())
        os.makedirs(self.directory_path, exist_ok=True)

    def remove_locks(self):
        """
        Removes the lock files since they are no longer required.
        """
        for file in os.listdir(self.directory_path):
            if file.endswith('.lock'):
                os.remove(file)

    def timestamp(self, fmt='%d\%m\%y_%H:%M:%S'):
        """
        Returns current timestamp.
        """
        return datetime.datetime.fromtimestamp(time.time()).strftime(fmt)

    def write_summary(self, key, observers):
        # Iterate through all observers.
        for observer in observers:
            observer_summary = observer.file_summary()
            # Iterate through all observations made by the observer.
            for observation in observer_summary:
                # Use observation name to get file.
                file_path = os.path.join(self.directory_path, key + '_' + observation[0])
                lock_path = file_path + '.lock'
                with FileLock(lock_path):
                    with open(file_path, 'a') as file:
                        # Write observation data to the last line.
                        file.write(observation[1])

    def summary_debug(self):
        file_path = os.path.join(self.directory_path, 'debug')
        lock_path = file_path + '.lock'
        with FileLock(lock_path):
            with open(file_path, 'a') as file:
                # Write observation data to the last line.
                file.write('This is a test\n')


####################################################################################################
####################################################################################################
####################################################################################################