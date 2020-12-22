# PyDeepImageJ

[![GitHub](https://img.shields.io/github/license/deepimagej/pydeepimagej)](https://raw.githubusercontent.com/deepimagej/pydeepimagej/master/LICENSE)
[![minimal Python version](https://img.shields.io/badge/Python-3-6666ff.svg)](https://www.anaconda.com/distribution/)

Python code to export trained models and read them in Fiji & ImageJ using DeepImageJ plugin
  - Creates a configuration class in python with all the information about the trained model needed for its correct use in Fiji & ImageJ.
  - Includes the metadata of an example image.
  - Includes all expected results and needed pre / post-processing routines.
  - See [DeepImageJ webpage](https://deepimagej.github.io/deepimagej/) for more information. 

### Requirements & Installation

- PyDeepImageJ requires Python 3 to run. 
- TensorFlow: It runs using the local installation of TensorFlow, i.e. the one corresponding to the trained model. However, DeepImageJ is only compatible with TensorFlow versions <= 2.2.1.


To install PyDeepImageJ either clone this repository or use PyPi via `pip`:

```sh
$ pip install pydeepimagej
```
or
```sh
$ git clone https://github.com/deepimagej/pydeepimagej.git
$ cd pydeepimagej
$ pip install .
```
----

### Reference: 
E. Gómez-de-Mariscal, C. García-López-de-Haro, L. Donati, M. Unser, A. Muñoz-Barrutia, D. Sage. 
*DeepImageJ: A user-friendly plugin to run deep learning models in ImageJ*, bioRxiv 2019
DOI: [https://doi.org/10.1101/799270](https://doi.org/10.1101/799270)
- Bioengineering and Aerospace Engineering Department, Universidad Carlos III de Madrid, Spain
- Biomedical Imaging Group, Ecole polytechnique federale de Lausanne (EPFL), Switzerland

Corresponding authors: mamunozb@ing.uc3m.es, daniel.sage@epfl.ch
Copyright © 2019. Universidad Carlos III, Madrid; Spain and EPFL, Lausanne, Switzerland.

#### License

[BSD 2-Clause License](https://raw.githubusercontent.com/deepimagej/pydeepimagej/master/LICENSE)

----

### Example of how to use it

Let `model` be a Keras or TensorFlow trained model. Initialize the configuration class with the trained model `model`
````python
from pydeepimagej.yaml import BioimageConfig
dij_config = BioimageConfig(model)
````
Update model information
````python
dij_config.Name = 'My trained model v0.1'
dij_config.Description = 'Brief description of the task to perform by the trained model'
dij_config.Authors = ['First author', 'Secon Author', 'Third Author who create the configuration specifications']
dij_config.References = ['Gómez-de-Mariscal, E., García-López-de-Haro, C. et al., bioRxiv 2019']
dij_config.DOI = ['https://doi.org/10.1101/799270']
dij_config.GitHub = ['https://github.com/deepimagej/pydeepimagej']
dij_config.Date = 'September-2020'
dij_config.License = 'BSD-3'
dij_config.Documentation = 'https://useful_documentation.pdf'
dij_config.Tags = ['deepimagej', 'segmentation', 'Fiji', 'microscopy']
dij_config.CoverImage =  ['./input.png', './output.png']
dij_config.Framework = 'TensorFlow'

````
**Prepare an ImageJ pre/post-processing macro.** 
You may need to preprocess the input image before the inference. Some ImageJ macro routines can be downloaded from [here](https://github.com/deepimagej/imagej-macros/) and included in the model specifications. Note that ImageJ macros are text files so it is easy to modify them inside a Python script ([see an example](https://github.com/deepimagej/pydeepimagej/blob/master/README.md#additional-commands)). To add any ImageJ macro code we need to run `add_preprocessing(local_path_to_the_macro_file, 'name_to_store_the_macro_in_the_bundled_model')`:
````python
path_preprocessing = "PercentileNormalization.ijm"`
# Download the macro file
urllib.request.urlretrieve("https://raw.githubusercontent.com/deepimagej/imagej-macros/master/PercentileNormalization.ijm", path_preprocessing )
# Include it in the configuration class
dij_config.add_preprocessing(path_preprocessing, "preprocessing")
````
The same holds for the postprocessing.
````python
path_postprocessing = "8bitBinarize.ijm"
urllib.request.urlretrieve("https://raw.githubusercontent.com/deepimagej/imagej-macros/master/8bitBinarize.ijm", path_postprocessing )
# Include the info about the postprocessing 
dij_config.add_postprocessing(path_postprocessing,"postprocessing")
````
DeepImageJ accepts two pre/post-processing routines. The images will be processed in the order in which we include them with `add_postprocessing`. Thus, in this example, the output of the model is first binarized with `'8bitBinarize.ijm'` and then, processed with `'another_macro.ijm'`: 
````python
path_second_postprocessing = './folder/another_macro.ijm'
dij_config.add_postprocessing(path_second_postprocessing, 'postprocessing_2')
````

**Add information about the example image.**
Let `test_img` be an example image to test the model inference and `test_prediction` be the resulting image after the post-processing. It is possible to export the trained model with these two, so an end user can see an example. 
`PixelSize` should be a list of values according to `test_img` dimensions and given in microns (µm). 
````python
PixelSize = [0.64,0.64,1] # Pixel size of a 3D volume with axis yxz
dij_config.add_test_info(test_img, test_prediction, PixelSize)
````

**Store weights using specific formats.**
The weights of a trained model can be stored either as a TensorFlow SavedModel bundle (`saved_model.pb` + `variables/`) or as a Keras HDF5 model (`model.h5`). Let `model` be a trained model in TensorFlow. With pydeepimagej, the weights information can be included as follows:
````python
dij_config.add_weights_formats(model, 'KerasHDF5', authors=['Authors', 'who', 'trained it'])
dij_config.add_weights_formats(model, 'TensorFlow', parent='keras_hdf5', authors=['Authors who', 'converted the model', 'into this new format'])
````
which will result in the `model.yaml` as:
````yaml
weights:
  keras_hdf5:
    source: ./keras_model.h5
    sha256: 9f7512eb28de4c6c4182f976dd8e53e9da7f342e14b2528ef897a970fb26875d
    authors:
    - Authors
    - who
    - trained it
  tensorflow_saved_model_bundle:
    source: ./tensorflow_saved_model_bundle.zip
    sha256: 2c552aa561c3c3c9063f42b78dda380e2b85a8ad04e434604af5cbb50eaaa54d
    parent: keras_hdf5
    authors:
    - Authors who
    - converted the model
    - into this new format
````

**EXPORT THE MODEL**

````python
deepimagej_model_path = './my_trained_model_deepimagej'
dij_config.export_model(deepimagej_model_path)
`````
When exporting the model a new folder with a DeepImageJ 2.1.0 bundled model is created. The folder is also provided as a zip file so it can be easily transferable.

### Additional commands
**Change one line in an ImageJ macro**
````
# Download the macro file
path_postprocessing = "8bitBinarize.ijm"
urllib.request.urlretrieve("https://raw.githubusercontent.com/deepimagej/imagej-macros/master/8bitBinarize.ijm", path_postprocessing )
# Modify the threshold in the macro to the chosen threshold
ijmacro = open(path_postprocessing,"r")  
list_of_lines = ijmacro. readlines()
# Line 21 is the one corresponding to the optimal threshold
list_of_lines[21] = "optimalThreshold = {}\n".format(128)
ijmacro.close()
ijmacro = open(path_postprocessing,"w")  
ijmacro. writelines(list_of_lines)
ijmacro. close()
````
**Estimation of the step size for the shape of the input image.**
If the model is an encoder-decoder with skip connections, and the input shape of your trained model is not fixed (i.e. `[None, None, 1]` ), the input shape still needs to fit some requirements. You can caluculate it knowing the number of poolings in the encoder path of the network:
````python
import numpy as np
pooling_steps = 0
for keras_layer in model.layers:
    if keras_layer.name.startswith('max') or "pool" in keras_layer.name:
      pooling_steps += 1
dij_config.MinimumSize = np.str(2**(pooling_steps))
````
### Exceptions
pydeepimagej is meant to connect Python with DeepImageJ so images can be processed in the Fiji/ImageJ ecosystem. Hence, images (tensors) are expected to have at least 3 dimensions: height, width and channels. For this reason, models with input shapes of less than 4 dimensions (`model.input_shape = [batch, height, width, channels]` are not considered. For example, if you have the following situation:
```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
```
please, modify it to
```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
```

### TODO list

 - Addapt pydeepimagej to PyTorch models so it can export trained models into TorchScript format.
 - Consider multiple inputs and outputs.
 - Include pre and post-processing routines as specified in the Bioimage.IO for the fields `inputs` and `outputs`.

