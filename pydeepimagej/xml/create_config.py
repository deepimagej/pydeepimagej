"""
DeepImageJ

https://deepimagej.github.io/deepimagej/

Conditions of use:

DeepImageJ is an open source software (OSS): you can redistribute it and/or modify it under 
the terms of the BSD 2-Clause License.

In addition, we strongly encourage you to include adequate citations and acknowledgments 
whenever you present or publish results that are based on it.
 
DeepImageJ is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
 
You should have received a copy of the BSD 2-Clause License along with DeepImageJ. 
If not, see <https://opensource.org/licenses/bsd-license.php>.


Reference: 
    
DeepImageJ: A user-friendly plugin to run deep learning models in ImageJ
E. Gomez-de-Mariscal, C. Garcia-Lopez-de-Haro, L. Donati, M. Unser, A. Munoz-Barrutia, D. Sage. 
Submitted 2019.

Bioengineering and Aerospace Engineering Department, Universidad Carlos III de Madrid, Spain
Biomedical Imaging Group, Ecole polytechnique federale de Lausanne (EPFL), Switzerland

Corresponding authors: mamunozb@ing.uc3m.es, daniel.sage@epfl.ch
 
Copyright 2019. Universidad Carlos III, Madrid, Spain and EPFL, Lausanne, Switzerland.

"""

import os
import xml.etree.ElementTree as ET
import time
import numpy as np
import urllib
import shutil
from skimage import io
from ..DeepImageJConfig import DeepImageJConfig

"""
Download the template from this link: 
    https://raw.githubusercontent.com/esgomezm/python4deepimagej/yaml/yaml/config_template.xml
TensorFlow library is needed. It is imported later to save the model as a SavedModel protobuffer

Try to check TensorFlow version and read DeepImageJ's compatibility requirements. 

import tensorflow as tf
tf.__version__
----------------------------------------------------
Example:
----------------------------------------------------
dij_config = xmlConfig(model)
# Update model information
dij_config.Authors = authors
dij_config.Credits = credits

# Add info about the minimum size in case it is not fixed.
pooling_steps = 0
for keras_layer in model.layers:
if keras_layer.name.startswith('max') or "pool" in keras_layer.name:
  pooling_steps += 1
dij_config.MinimumSize = np.str(2**(pooling_steps))

# Add the information about the test image
dij_config.add_test_info(test_img, test_prediction, PixelSize)

## Prepare preprocessing file
path_preprocessing = "PercentileNormalization.ijm"
urllib.request.urlretrieve("https://raw.githubusercontent.com/deepimagej/imagej-macros/master/PercentileNormalization.ijm", path_preprocessing )
# Include the info about the preprocessing 
dij_config.add_preprocessing(path_preprocessing, "preprocessing")

## Prepare postprocessing file
path_postprocessing = "8bitBinarize.ijm"
urllib.request.urlretrieve("https://raw.githubusercontent.com/deepimagej/imagej-macros/master/8bitBinarize.ijm", path_postprocessing )
# Include the info about the postprocessing 
post_processing_name =  "postprocessing_LocalMaximaSMLM"
dij_config.add_postprocessing(path_postprocessing_max,post_processing_name)

## EXPORT THE MODEL
deepimagej_model_path = os.path.join(QC_model_folder, 'deepimagej')
dij_config.export_model(model, deepimagej_model_path)
----------------------------------------------------
Example: change one line in an ImageJ macro
----------------------------------------------------
## Prepare postprocessing file
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
"""
class xmlConfig(DeepImageJConfig):
    # Import all the information needed for DeepImageJ
    def __init__(self, tf_model):
        DeepImageJConfig.__init__(self, tf_model)
    
        self.get_dimensions(tf_model)
        # Receptive field of the network to process input
        self.Padding = np.str(self._pixel_half_receptive_field(tf_model))

    def get_dimensions(self, tf_model):
        """
        Calculates the array organization and shapes of inputs and outputs.
        """
        input_dim = tf_model.input_shape
        output_dim = tf_model.output_shape
        # Deal with the order of the dimensions and whether the size is fixed
        # or not
        if input_dim[2] is None:
            self.FixedPatch = 'false'
            self.PatchSize = self.MinimumSize
            if input_dim[-1] is None:
                self.InputOrganization0 = 'NCHW'
                self.Channels = np.str(input_dim[1])
            else:
                self.InputOrganization0 = 'NHWC'
                self.Channels = np.str(input_dim[-1])

            if output_dim[-1] is None:
                self.OutputOrganization0 = 'NCHW'
            else:
                self.OutputOrganization0 = 'NHWC'
        else:
            self.FixedPatch = 'true'
            self.PatchSize = np.str(input_dim[2])

            if input_dim[-1] < input_dim[-2] and input_dim[-1] < input_dim[-3]:
                self.InputOrganization0 = 'NHWC'
                self.Channels = np.str(input_dim[-1])
            else:
                self.InputOrganization0 = 'NCHW'
                self.Channels = np.str(input_dim[1])

            if output_dim[-1] < output_dim[-2] and output_dim[-1] < output_dim[-3]:
                self.OutputOrganization0 = 'NHWC'
            else:
                self.OutputOrganization0 = 'NCHW'

        # Adapt the format from brackets to parenthesis
        input_dim = np.str(input_dim)
        input_dim = input_dim.replace('(', ',')
        input_dim = input_dim.replace(')', ',')
        input_dim = input_dim.replace('None', '-1')
        input_dim = input_dim.replace(' ', "")
        self.InputTensorDimensions = input_dim

    def _pixel_half_receptive_field(self, tf_model):
        """
        The halo is equivalent to the receptive field of one pixel. This value
        is used for image reconstruction when a entire image is processed.
        """
        input_shape = tf_model.input_shape

        if self.FixedPatch == 'false':
            min_size = 50 * np.int(self.MinimumSize)

            if self.InputOrganization0 == 'NHWC':
                null_im = np.zeros((1, min_size, min_size, input_shape[-1])
                                   , dtype=np.float32)
            else:
                null_im = np.zeros((1, input_shape[1], min_size, min_size)
                                   , dtype=np.float32)
        else:
            null_im = np.zeros((input_shape[1:])
                               , dtype=np.float32)
            null_im = np.expand_dims(null_im, axis=0)
            min_size = np.int(self.PatchSize)

        point_im = np.zeros_like(null_im)
        min_size = np.int(min_size / 2)

        if self.InputOrganization0 == 'NHWC':
            point_im[0, min_size, min_size] = 1
        else:
            point_im[0, :, min_size, min_size] = 1

        result_unit = tf_model.predict(np.concatenate((null_im, point_im)))

        D = np.abs(result_unit[0] - result_unit[1]) > 0

        if self.InputOrganization0 == 'NHWC':
            D = D[:, :, 0]
        else:
            D = D[0, :, :]

        ind = np.where(D[:min_size, :min_size] == 1)
        halo = np.min(ind[1])
        halo = min_size - halo + 1

        return halo

    def export_model(self, tf_model,deepimagej_model_path, **kwargs):
        """
        Main function to export the model as a bundled model of DeepImageJ
        tf_model:              tensorflow/keras model
        deepimagej_model_path: directory where DeepImageJ model is stored.
        """
        # Save the mode as protobuffer
        self.save_tensorflow_pb(tf_model, deepimagej_model_path)

        # extract the information about the testing image
        test_info = self.test_info
        io.imsave(os.path.join(deepimagej_model_path,'exampleImage.tiff'), self.test_info.InputImage)
        io.imsave(os.path.join(deepimagej_model_path,'resultImage.tiff'), self.test_info.OutputImage)
        print("Example images stored.")

        # write the DeepImageJ configuration as an xml file
        write_config(self, test_info, deepimagej_model_path)
        
        # Add preprocessing and postprocessing macros. 
        # More than one is available, but the first one is set by default.
        for i in range(len(self.Preprocessing)):
          shutil.copy2(self.Preprocessing_files[i], os.path.join(deepimagej_model_path, self.Preprocessing[i]))
          print("ImageJ macro {} included in the bundled model.".format(self.Preprocessing[i]))

        for i in range(len(self.Postprocessing)):
          shutil.copy2(self.Postprocessing_files[i], os.path.join(deepimagej_model_path, self.Postprocessing[i]))
          print("ImageJ macro {} included in the bundled model.".format(self.Postprocessing[i]))

        # Zip the bundled model to download
        shutil.make_archive(deepimagej_model_path, 'zip', deepimagej_model_path)
        print("DeepImageJ model was successfully exported as {0}.zip. You can download and start using it in DeepImageJ.".format(deepimagej_model_path))
          

    def save_tensorflow_pb(self,tf_model, deepimagej_model_path):
        # Check whether the folder to save the DeepImageJ bundled model exists.
        # If so, it needs to be removed (TensorFlow requirements)
        # -------------- Other definitions -----------
        W  = '\033[0m'  # white (normal)
        R  = '\033[31m' # red
        if os.path.exists(deepimagej_model_path):
            print(R+'!! WARNING: DeepImageJ model folder already existed and has been removed !!'+W)
            shutil.rmtree(deepimagej_model_path)

        import tensorflow as tf
        TF_VERSION = tf.__version__
        print("DeepImageJ model will be exported using TensorFlow version {0}".format(TF_VERSION))
        if TF_VERSION[:3] == "2.3":
            print(R+"DeepImageJ plugin is only compatible with TensorFlow version 1.x, 2.0.0, 2.1.0 and 2.2.0. Later versions are not suported in DeepImageJ."+W)
        
        def _save_model():
            if tf_version==2:
                """TODO: change it once TF 2.3.0 is available in JAVA"""
                from tensorflow.compat.v1 import saved_model
                from tensorflow.compat.v1.keras.backend import get_session
            else:
                from tensorflow import saved_model
                from keras.backend import get_session

            builder = saved_model.builder.SavedModelBuilder(deepimagej_model_path)

            signature = saved_model.signature_def_utils.predict_signature_def(
                                              inputs  = {'input':  tf_model.input},
                                              outputs = {'output': tf_model.output} )
            
            signature_def_map = { saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature }
            
            builder.add_meta_graph_and_variables( get_session(),
                                                  [saved_model.tag_constants.SERVING], 
                                                  signature_def_map=signature_def_map )
            builder.save()
            print("TensorFlow model exported to {0}".format(deepimagej_model_path))

        if TF_VERSION[0] == '1':
            tf_version = 1
            _save_model()
        else:
            tf_version = 2
            """TODO: change it once TF 2.3.0 is available in JAVA"""
            from tensorflow.keras.models import clone_model
            _weights = tf_model.get_weights()
            with tf.Graph().as_default():
                # clone model in new graph and set weights
                _model = clone_model(tf_model)
                _model.set_weights(_weights)
                _save_model()


def write_config(Config, TestInfo, config_path):
    """
    - Config:       Class with all the information about the model's architecture and pre/post-processing
    - TestInfo:   Metadata of the image provided as an example
    - config_path:  path to the template of the configuration file. 
    It can be downloaded from: 
      https://raw.githubusercontent.com/deepimagej/python4deepimagej/blob/master/xml/config_template.xml
    The function updates the fields in the template provided with the
    information about the model and the example image.
    """
    urllib.request.urlretrieve("https://raw.githubusercontent.com/deepimagej/python4deepimagej/master/xml/config_template.xml", "config_template.xml")
    try:
        tree = ET.parse('config_template.xml')
        root = tree.getroot()
    except:
        print("config_template.xml not found.")
    
    # WorkCitation-Credits
    root[0][0].text = Config.Name
    root[0][1].text = Config.Authors
    root[0][2].text = Config.URL
    root[0][3].text = Config.Credits
    root[0][4].text = Config.Version
    root[0][5].text = Config.Date
    root[0][6].text = Config.References
    
    # ExampleImage
    root[1][0].text = TestInfo.Input_shape
    root[1][1].text = TestInfo.Output_shape
    root[1][2].text = TestInfo.MemoryPeak
    root[1][3].text = TestInfo.Runtime
    root[1][4].text = TestInfo.PixelSize
    
    # ModelArchitecture
    root[2][0].text = 'tf.saved_model.tag_constants.SERVING'
    root[2][1].text = 'tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY'
    root[2][2].text = Config.InputTensorDimensions
    root[2][3].text = '1'
    root[2][4].text = 'input'
    root[2][5].text = Config.InputOrganization0
    root[2][6].text = '1'
    root[2][7].text = 'output'
    root[2][8].text = Config.OutputOrganization0
    root[2][9].text = Config.Channels
    root[2][10].text = Config.FixedPatch
    root[2][11].text = Config.MinimumSize
    root[2][12].text = Config.PatchSize
    root[2][13].text = 'true'
    root[2][14].text = Config.Padding
    root[2][15].text = Config.Preprocessing[0]
    print("Preprocessing macro '{}' set by default".format(Config.Preprocessing[0]))
    root[2][16].text = Config.Postprocessing[0]
    print("Postprocessing macro '{}' set by default".format(Config.Postprocessing[0]))
    root[2][17].text = '1'    
    try:
        tree.write(os.path.join(config_path,'config.xml'),encoding="UTF-8",xml_declaration=True, )
        print("DeepImageJ configuration file exported.")
    except:
        print("The directory {} does not exist.".format(config_path))
