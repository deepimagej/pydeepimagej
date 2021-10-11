# PydeepImageJ

[![GitHub](https://img.shields.io/github/license/deepimagej/pydeepimagej)](https://raw.githubusercontent.com/deepimagej/pydeepimagej/master/LICENSE)
[![minimal Python version](https://img.shields.io/badge/Python-3-6666ff.svg)](https://www.anaconda.com/distribution/)

Python code to export trained models into the [BioImage Model Zoo](https://bioimage.io/) format and read them in Fiji & ImageJ using the deepImageJ plugin.
  - Creates a configuration class in Python with all the information about the trained model needed for its correct use in Fiji & ImageJ.
  - Includes the metadata of an example image.
  - Includes all expected results and needed pre / post-processing routines.
  - Creates basic cover images for the model card in the BioImage Model Zoo.
  - Creates de the version 0.3.2 of the [BioImage Model Zoo specification file](https://bioimage.io/docs/#/contribute_models/README?id=model-contribution-requirements): `model.yaml`   
  - See [deepImageJ webpage](https://deepimagej.github.io/deepimagej/) for more information about how to use the model in Fiji & ImageJ. 

### Requirements & Installation

- PyDeepImageJ requires Python 3 to run. 
- TensorFlow: It runs using the local installation of TensorFlow, i.e. the one corresponding to the trained model. However, deepImageJ is only compatible with TensorFlow versions <= 2.2.1.

To install pydeepImageJ either clone this repository or use PyPi via `pip`:

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
* Gómez-de-Mariscal, E., García-López-de-Haro, C., Ouyang, W., Donati, L., Lundberg, L., Unser, M., Muñoz-Barrutia, A. and Sage, D., "DeepImageJ: A user-friendly environment to run deep learning models in ImageJ", Nat Methods 18, 1192–1195 (2021). 
https://doi.org/10.1038/s41592-021-01262-9
  * **Read the paper online with this link: [rdcu.be/cyG3K](https://rdcu.be/cyG3K)**

- Bioengineering and Aerospace Engineering Department, Universidad Carlos III de Madrid, Spain
- Science for Life Laboratory, KTH – Royal Institute of Technology, Stockholm, Sweden
- Biomedical Imaging Group, Ecole polytechnique federale de Lausanne (EPFL), Switzerland

Corresponding authors: mamunozb@ing.uc3m.es, daniel.sage@epfl.ch
Copyright © 2019. Universidad Carlos III, Madrid; Spain and EPFL, Lausanne, Switzerland.
#### How to cite
```bibtex
@article{gomez2021deepimagej,
  title={DeepImageJ: A user-friendly environment to run deep learning models in ImageJ},
  author={G{\'o}mez-de-Mariscal, Estibaliz and Garc{\'i}a-L{\'o}pez-de-Haro, Carlos and Ouyang, Wei and Donati, Laur{\`e}ne and Lundberg, Emma and Unser, Michael and Mu{\~{n}}oz-Barrutia, Arrate and Sage, Daniel},
  journal={Nature Methods},
  year={2021},
  volume={18},
  number={10},
  pages={1192-1195},
  URL = {https://doi.org/10.1038/s41592-021-01262-9},
  doi = {10.1038/s41592-021-01262-9}
}
```
#### License

[BSD 2-Clause License](https://raw.githubusercontent.com/deepimagej/pydeepimagej/master/LICENSE)

----

## Example of how to use it
Try a Jupyter notebook in Google Colaboratory: [![GoogleColab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepimagej/pydeepimagej/blob/master/examples/ExportBioImageModelZoo_deepImageJ.ipynb)

Otherwise, follow the next steps:

Let `model` be a Keras or TensorFlow trained model. Initialize the configuration class with the trained model `model`
````python
from pydeepimagej.yaml import BioImageModelZooConfig
# MinimumSize needs to be given as it cannot be always estimated. See Additional commands for hints.
dij_config = BioImageModelZooConfig(model, MinimumSize)
````
Update model information
````python
dij_config.Name = 'My trained model v0.1'
dij_config.Description = 'Brief description of the task to perform by the trained model'
dij_config.Authors.Names = ['First author', 'Second Author', 'Third Author who create the configuration specifications']
dij_config.Authors.Affiliations = ['First author affiliation', 'Second author affiliation', 'Third author affiliation']
dij_config.References = ['Gómez-de-Mariscal, E., García-López-de-Haro, C. et al., bioRxiv 2019', 'Second citation']
dij_config.DOI = ['https://doi.org/10.1101/799270', 'second citation doi']
dij_config.GitHub = 'https://github.com/deepimagej/pydeepimagej'
dij_config.License = 'BSD-3'
dij_config.Documentation = 'https://useful_documentation.pdf'
dij_config.Tags = ['deepimagej', 'segmentation', 'Fiji', 'microscopy']
dij_config.CoverImage =  ['./input.png', './output.png']
dij_config.Framework = 'TensorFlow'
# Parent model in the BioImage Model Zoo whose trained weights were used as pretrained weights.
dij_config.Parent = "https://bioimage.io/#/?id=deepimagej%2FUNet2DPancreaticSegmentation"
````
### 1. Pre & post-processing specification.
#### 1.1. Specify the pre&post-processing steps following the BioImage Model Zoo specifications.
If the pre-processing or the post-processing can be defined using the implementations defined at
, then it is also possible to specify them with some code:
```python
dij_config.add_bioimageio_spec('pre-processing', 'scale_range',
                               mode='per_sample', axes='xyzc',
                               min_percentile=0, 
                               max_percentile=100)

dij_config.add_bioimageio_spec('post-processing', 'binarize',
                               threshold=threshold)
```
The `BioImageModelZooConfig` class will include as many steps as times the previous functions are called. For example:
```python
# Make sure that there's no pre-processing specified.
dij_config.BioImage_Preprocessing=None
dij_config.add_bioimageio_spec('pre-processing', 'scale_range',
                               mode='per_sample', axes='xyzc',
                               min_percentile=min_percentile, 
                               max_percentile=max_percentile)
dij_config.add_bioimageio_spec('pre-processing', 'scale_linear',
                               gain=255, offset=0, axes='xy')
```
```
dij_config.BioImage_Preprocessing:
[{'scale_range': {'kwargs': {'axes': 'xyzc',
  'max_percentile': 100,
  'min_percentile': 0,
  'mode': 'per_sample'}}},
 {'scale_range': {'kwargs': {'axes': 'xy', 'gain': 255, 'offset': 0}}}]
```
The same applies for the post-processing:
```python
dij_config.BioImage_Postprocessing=None 
dij_config.add_bioimageio_spec('post-processing', 'scale_range',
                               mode='per_sample', axes='xyzc', 
                               min_percentile=0, max_percentile=100)

dij_config.add_bioimageio_spec('post-processing', 'scale_linear',
                               gain=255, offset=0, axes='xy')

dij_config.add_bioimageio_spec('post-processing', 'binarize',
                               threshold=threshold)
```
```
dij_config.BioImage_Postprocessing:
[{'scale_range': {'kwargs': {'axes': 'xyzc',
  'max_percentile': 100,
  'min_percentile': 0,
  'mode': 'per_sample'}}},
 {'scale_range': {'kwargs': {'axes': 'xy', 'gain': 255, 'offset': 0}}},
 {'binarize': {'kwargs': {'threshold': 0.5}}}]
```
#### 1.2. Prepare an ImageJ pre/post-processing macro.
You may need to preprocess the input image before the inference. Some ImageJ macro routines can be downloaded from [here](https://github.com/deepimagej/imagej-macros/) and included in the model specifications. Note that ImageJ macros are text files so it is easy to modify them inside a Python script ([see an example](https://github.com/deepimagej/pydeepimagej/blob/master/README.md#additional-commands)). To add any ImageJ macro code we need to run `add_preprocessing(local_path_to_the_macro_file, 'name_to_store_the_macro_in_the_bundled_model')`:
````python
path_preprocessing = "PercentileNormalization.ijm"
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

### 2. Add information about the example image.
Let `test_img` be an example image to test the model inference and `test_prediction` be the resulting image after the post-processing. It is possible to export the trained model with these two, so an end user can see an example. 
`PixelSize` should be a list of values according to `test_img` dimensions and given in microns (µm). 
````python
PixelSize = [0.64,0.64,1] # Pixel size of a 3D volume with axis yxz
dij_config.add_test_info(test_img, test_prediction, PixelSize)
````

#### 2.1. Create some covers for the model card in the BioImage Model Zoo.
Let `test_img` and `test_mask` be the input and output example images, and `./input.png` and `./output.png` the names we want to use to store them within bundled model. `dij_config` stretches the intensity range of the given images to the [0, 255] range so the images can be exported as 8-bits images and visualized properly on the website.  
```python
dij_config.create_covers([test_img, test_mask])
dij_config.Covers =  ['./input.png', './output.png']
```

### 3. Store weights using specific formats.
The weights of a trained model can be stored either as a TensorFlow SavedModel bundle (`saved_model.pb` + `variables/`) or as a Keras HDF5 model (`model.h5`). Let `model` be a trained model in TensorFlow. With pydeepimagej, the weights information can be included as follows:
````python

dij_config.add_weights_formats(model, 'KerasHDF5', 
                               authors=['Authors', 'who', 'trained it'])
dij_config.add_weights_formats(model, 'TensorFlow', 
                               parent="keras_hdf5",
                               authors=['Authors who', 'converted the model', 'into this new format'])
````
which in the `model.yaml` appear as :
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

### 4. EXPORT THE MODEL
````python
deepimagej_model_path = './my_trained_model_deepimagej'
dij_config.export_model(deepimagej_model_path)
`````
When exporting the model, a new folder with a deepImageJ 2.1.0 bundled model is created. The folder is also provided as a zip file, so it can be easily transferable.

## Additional commands
### Change one line in an ImageJ macro
````python
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
### Estimation of the step size for the shape of the input image.
If the model is an encoder-decoder with skip connections, and the input shape of your trained model is not fixed (i.e. `[None, None, 1]` ), the input shape still needs to fit some requirements. You can caluculate it knowing the number of poolings in the encoder path of the network:
````python
import numpy as np
pooling_steps = 0
for keras_layer in model.layers:
    if keras_layer.name.startswith('max') or "pool" in keras_layer.name:
      pooling_steps += 1
MinimumSize = np.str(2**(pooling_steps))
````
## Exceptions
pydeepimagej is meant to connect Python with DeepImageJ so images can be processed in the Fiji & ImageJ ecosystem. Hence, images (tensors) are expected to have at least 3 dimensions: height, width and channels. For this reason, models with input shapes of less than 4 dimensions (`model.input_shape = [batch, height, width, channels]` are not considered. For example, if you have the following situation:
```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)])
```
please, modify it to
```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)])
```
## Code references used in this package:
This code uses similar functions to the ones in [StarDist](https://github.com/stardist/stardist) package for the calculation of a pixel's receptive field in a network. Citations:
- Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers.
  Cell Detection with Star-convex Polygons.
  International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Granada, Spain, September 2018.
  DOI: [10.1007/978-3-030-00934-2_30](https://doi.org/10.1007/978-3-030-00934-2_30)

- Martin Weigert, Uwe Schmidt, Robert Haase, Ko Sugawara, and Gene Myers.
  Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy.
  The IEEE Winter Conference on Applications of Computer Vision (WACV), Snowmass Village, Colorado, March 2020 
  DOI: [10.1109/WACV45572.2020.9093435](https://doi.org/10.1109/WACV45572.2020.9093435)
  
## TODO list

 - Addapt pydeepimagej to PyTorch models so it can export trained models into TorchScript format.
 - Consider multiple inputs and outputs.

