"""
This file loads MNIST dataset from the tensorflow datasets prefetchds loader. The easiest way to load the dataset.
## I like working in OOP design because I know at each step what are the I/Os , and use them for anything later. 
"""

from config.dp_utils import tfds, PREFETCH_MNIST_KWARGS_TFDSLOADER

class TFDSPrefetchMnist: 
    """
    The class for generating the MNIST dataset from tensorflow datasets
    """
    def __init__(self):
        # Private variables 
        self.__training_dataset = None # Prefetch style 
        self.__validation_dataset = None # Prefetch style
        self.__testing_dataset = None # Prefetch style 
        self.__dataset_info = None 

        # Public variables 

        # The Init Pipeline 
        self.__build_dataset()

    """
    The Init Pipeline 
    """

    def __build_dataset(self) : 
        """
        Building the dataset, from the config file
        """
        (self.__training_dataset,self.__validation_dataset,self.__testing_dataset),self.__dataset_info =  tfds.load(**PREFETCH_MNIST_KWARGS_TFDSLOADER) 

    
    """
    Getters
    """

    def get_training_dataset(self) : 
        return self.__training_dataset
    
    def get_validation_dataset(self) : 
        return self.__validation_dataset
    
    def get_testing_dataset(self) : 
        return self.__testing_dataset
