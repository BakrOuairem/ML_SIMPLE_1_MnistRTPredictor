# IMPORTS

import tensorflow_datasets as tfds
import tensorflow as tf

# CONSTANTS (Maybe better as a JSON ?)

PREFETCH_MNIST_KWARGS_TFDSLOADER = {
    "name" : "mnist", # Should not change 
    "split" : ['train+test[:70%]', 'train+test[70%:80%]','train+test[80%:]'],  # We merge the test and train. 
    "data_dir" : "__tmp_files/dataset", # To avoid any loaded dataset somewhere in the computer that may be never accessed again (memory optimisation)
    "batch_size" : 32 , # Should be good enough for every hardware 
    "shuffle_files" : True, 
    "download" : True, # Do not put False here because download and prepare function sees if the data already exists as well . 
    "as_supervised" : True,  # No point in putting to false (false will just transform the tuple to a dict with image and label keys)
    "with_info" : True, # No point in putting to false either.
}

EXPECTED_IMAGE_SIZE = [28,28]

help(tfds.load)