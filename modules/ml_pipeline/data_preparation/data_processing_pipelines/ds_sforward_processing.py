"""
This file defines a series of functions to process the MNIST dataset using TensorFlow. 
It includes functions for resizing, normalizing images, one-hot encoding labels, and preparing the dataset in question. 
The goal is to run this class at training and serving. 
A better design would be to have a parent class and then have children classes for each dataset.
"""

from config.utils import tf, EXPECTED_IMAGE_SIZE

class DSForwardProcessing:
    """
    The class for processing the MNIST dataset
    """
    def __init__(self):
        # Private variables
        self.__dataset : tf.data.Dataset = None # The dataset to be processed
    
        # Public variables
        self.processed_dataset : tf.data.Dataset  = None # The processed dataset
    
    """
    Pipeline functions 
    """

    # Sub functions 
    def __resize_image(self, image, label):
        """
        Resizing the image to 28x28, works by stretching the image as well. 
        """
        image = tf.image.resize(image, EXPECTED_IMAGE_SIZE)
        return image, label

    def __normalize_image(self, image, label):
        """
        Normalizing the image to [0,1], for better training performance
        0-255 -> 0.0-1.0
        """
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    def __one_hot_encode_label(self, image, label):
        """
        One hot encoding the label 
        """
        label = tf.one_hot(label, depth=10) # 10 classes in MNIST (0-9)
        return image, label
    
    # Main function
    def process(self, dataset): # Returns the processed dataset
        """
        Preparing the dataset by applying the processing functions
        The pipeline is as follows:
        1. Resize the image to 28x28 : Standard size for MNIST
        2. Normalize the image to [0,1] : Better for training performance
        3. One hot encode the label : Training performance for classification
        4. Cache the dataset : This is to avoid recomputing the transformations every epoch 
        5. Prefetch the dataset for better performance : This is to overlap the preprocessing and model execution of data.
        6. Return the processed dataset
        """
        # First we map the function to resize the image, the number of parallel calls is set to AUTOTUNE to optimize the performance. 
        # tf.data.AUTOTUNE means that the number of parallel calls will be determined at runtime based on the available CPU resources.
        self.__dataset = dataset # Last dataset to be processed 
        self.processed_dataset = self.__dataset.map(self.__resize_image, num_parallel_calls=tf.data.AUTOTUNE)
        self.processed_dataset = self.processed_dataset.map(self.__normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
        self.processed_dataset = self.processed_dataset.map(self.__one_hot_encode_label, num_parallel_calls=tf.data.AUTOTUNE)
        self.processed_dataset = self.processed_dataset.cache()
        self.processed_dataset = self.processed_dataset.prefetch(tf.data.AUTOTUNE)
        return self.processed_dataset
    
    """
    Reversing functions 
    """

    def __reverse_one_hot_encode_label(self, image,  one_hot_label):
        """
        Reversing the one hot encoding of the label 
        """
        label = tf.argmax(one_hot_label, axis=-1)
        return image,  label
    
    def __reverse_normalize_image(self, normalized_image, label):
        """
        Reversing the normalization of the image 
        """
        image = normalized_image * 255
        return image, label
    
    def __reverse_resize_image(self, resized_image, label , original_size):
        """
        Reversing the resizing of the image 
        """
        image = tf.cast(tf.image.resize(resized_image, original_size), tf.uint8) 
        return image, label 
    
    @tf.autograph.experimental.do_not_convert
    def reverse_process(self, dataset :tf.data.Dataset, original_size): 
        """
        Reversing the processing of the dataset
        The pipeline is as follows:
        1. Reverse the one hot encoding of the label : Get back the original label
        2. Reverse the normalization of the image : Get back the original image
        3. Reverse the resizing of the image : Get back the original image size
        4. Return the reversed dataset
        """
        reversed_dataset = dataset.map(self.__reverse_one_hot_encode_label, num_parallel_calls=tf.data.AUTOTUNE)
        reversed_dataset = reversed_dataset.map(self.__reverse_normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
        reversed_dataset = reversed_dataset.map(lambda img, lbl: self.__reverse_resize_image(img, lbl, original_size), num_parallel_calls=tf.data.AUTOTUNE)
        reversed_dataset = reversed_dataset.prefetch(tf.data.AUTOTUNE)
        return reversed_dataset
    
    """
    Getters 
    """

    def get_last_processed_dataset(self) : 
        return self.processed_dataset
    
    def get_last_raw_dataset(self) : 
        return self.__dataset