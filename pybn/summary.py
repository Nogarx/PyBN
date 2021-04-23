import datetime
import time
import os
import numpy as np
import random 
import ray

####################################################################################################
####################################################################################################
####################################################################################################

@ray.remote
class SummaryWritter():

    def __init__(self, configuration):
        self.storage_path = configuration['storage_path']
        self.directory_path = None
        self.file_path = None
        self.queue = ray.util.queue.Queue()
        self.initialize_directory()

    def initialize_directory(self):
        """
        Create a new directory to store the data of the current experiment.
        """
        self.directory_path = os.path.join(self.storage_path, self.timestamp())
        os.makedirs(self.directory_path, exist_ok=True)

    def create_file(self, name):
        """
        Creates a new file, within the SummaryWriter's directory, to store the data of the current experiment.
        """
        file_path = os.path.join(self.directory_path, name)
        file = open(file_path, 'x')
        self.file_path = file_path

    def timestamp(self, fmt='%y%m%dT%H%M%S'):
        """
        Returns current timestamp.
        """
        return datetime.datetime.fromtimestamp(time.time()).strftime(fmt)

    def write_summary(self):
        with open(self.file_path, 'w') as file:
            while True:
                data = self.queue.get(block=True)
                file.write(data)


####################################################################################################
####################################################################################################
####################################################################################################