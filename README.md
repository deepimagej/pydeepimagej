# PyDeepImageJ

[![GitHub](https://img.shields.io/github/license/deepimagej/pydeepimagej)](https://raw.githubusercontent.com/deepimagej/pydeepimagej/master/LICENSE)
[![minimal Python version](https://img.shields.io/badge/Python-3-6666ff.svg)](https://www.anaconda.com/distribution/)

Python code to export trained models and read them in Fiji & ImageJ using DeepImageJ plugin
  - Create a configuration class in python with all the information about the trained model, needed for its correct use in Fiji & ImageJ.
  - Include the metadata of an example image.
  - Include all expected results and needed pre / post-processing routines.
  - See [DeepImageJ webpage](https://deepimagej.github.io/deepimagej/) for more information. 

### Installation
**Requirements**
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
$ python setup.py install
```

### Example of how to use it

Let `model` be a Keras or TensorFlow trained model. Initialize the configuration class with the trained model `model`
````
dij_config = DeepImageJConfig(model)
````
Update model information
````
dij_config.Name = 'My trained model v0.1'
dij_config.Authors = 'First author; Secon Author; Third Author'
dij_config.Credits = 'Institution 1; Institution 2; Institution 3'
dij_config.URL        = 'https://github.com/user/your_model'
dij_config.Version    = '0.1'
dij_config.References = 'C. García-López-de-Haro et al., bioRxiv 2019'
dij_config.Date       = 'September-2020'
````
**Prepare an ImageJ pre/post-processing macro.** 
You may need to preprocess the input image before the inference. Some ImageJ macro routines can be downloaded from [here](https://github.com/deepimagej/imagej-macros/) and included in the model specifications. Note that ImageJ macros are text files so it is easy to modify them inside a Python script ([see an example](https://github.com/deepimagej/pydeepimagej/blob/master/README.md#additional-commands)):
````
path_preprocessing = "PercentileNormalization.ijm"`
# Download the macro file
urllib.request.urlretrieve("https://raw.githubusercontent.com/deepimagej/imagej-macros/master/PercentileNormalization.ijm", path_preprocessing )
# Include it in the configuration class
dij_config.add_preprocessing(path_preprocessing, "preprocessing")
````
The same holds for the postprocessing.
````
path_postprocessing = "8bitBinarize.ijm"
urllib.request.urlretrieve("https://raw.githubusercontent.com/deepimagej/imagej-macros/master/8bitBinarize.ijm", path_postprocessing )
# Include the info about the postprocessing 
dij_config.add_postprocessing(path_postprocessing,"postprocessing")
````
It is even possible to include different pre/post-processing routines in the same model as long as their names always start with "preprocessing" or "postprocessing" respectively: 
````
path_second_postprocessing = "./folder/another_macro.ijm"
dij_config.add_postprocessing(path_second_postprocessing, "postprocessing_2")
````

**Add information about the example image.**
Let `test_img` be an example image to test the model inference and `test_prediction` be the resulting image after the post-processing. It is possible to export the trained model with these two, so an end user can see an example. 
`PixelSize` should be given in microns (µm). 
````
dij_config.add_test_info(test_img, test_prediction, PixelSize)
````

**EXPORT THE MODEL**
````
deepimagej_model_path = './my_trained_model_deepimagej'
dij_config.export_model(model, deepimagej_model_path)
`````
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
````
import numpy as np
pooling_steps = 0
for keras_layer in model.layers:
    if keras_layer.name.startswith('max') or "pool" in keras_layer.name:
      pooling_steps += 1
dij_config.MinimumSize = np.str(2**(pooling_steps))
````
### Todo list

 - Write bioimage.IO config.yaml class

Reference: 
----
E. Gomez-de-Mariscal, C. Garcia-Lopez-de-Haro, L. Donati, M. Unser, A. Munoz-Barrutia, D. Sage. 
*DeepImageJ: A user-friendly plugin to run deep learning models in ImageJ*, bioRxiv 2019
DOI: [https://doi.org/10.1101/799270](https://doi.org/10.1101/799270)
- Bioengineering and Aerospace Engineering Department, Universidad Carlos III de Madrid, Spain
- Biomedical Imaging Group, Ecole polytechnique federale de Lausanne (EPFL), Switzerland

Corresponding authors: mamunozb@ing.uc3m.es, daniel.sage@epfl.ch
Copyright © 2019. Universidad Carlos III, Madrid; Spain and EPFL, Lausanne, Switzerland.

License
----
[BSD 2-Clause License](https://raw.githubusercontent.com/deepimagej/pydeepimagej/master/LICENSE)
