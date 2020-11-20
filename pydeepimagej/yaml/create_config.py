import os
import xml.etree.ElementTree as ET
import time
import numpy as np
import urllib
import shutil
from skimage import io
import yaml
from DeepImageJConfig import DeepImageJConfig


class BioimageConfig:
    def __init__(self, tf_model):
        # Import all the information needed for DeepImageJ
        DeepImageJConfig.__init__(self, tf_model)
        # New fields for the Bioimage.IO configuration file
        self.Description = 'null'
        self.DOI = 'null'
        self.Documentation = 'null'
        self.Format_version = 'null'
        self.License = 'BSD-3'
        self.Source = 'null'
        self.Tags = ['deepimagej']
        self.WeightsVersion = 'v1'
        self.get_dimensions(tf_model)
        # Receptive field of the network to process input
        if self.OutputOrganization0 is not 'list':
            self.Padding = np.str(self._pixel_half_receptive_field(tf_model))

        self.ModelInput = tf_model.input_shape
        self.ModelOutput = tf_model.output_shape
        self.OutputOffset = '[0,0,0,0]'
        self.OutputScale = '[1,1,1,1]'

    def get_dimensions(self, tf_model):
        """
        Calculates the array organization and shapes of inputs and outputs.
        """
        input_dim = tf_model.input_shape
        output_dim = tf_model.output_shape
        if len(output_dim) < 4:
            self.OutputOrganization0 = 'list'
        # Deal with the order of the dimensions and whether the size is fixed
        # or not
        if input_dim[2] is None:
            self.FixedPatch = 'false'
            self.PatchSize = self.MinimumSize
            if len(input_dim)==4:
                if input_dim[-1] is None:
                    self.InputOrganization0 = 'bcyx'
                    self.Channels = np.str(input_dim[1])
                else:
                    self.InputOrganization0 = 'byxc'
                    self.Channels = np.str(input_dim[-1])
            elif len(input_dim)==5:
                if input_dim[-1] is None:
                    self.InputOrganization0 = 'bcyxz'
                    self.Channels = np.str(input_dim[1])
                else:
                    self.InputOrganization0 = 'byxcz'
                    self.Channels = np.str(input_dim[-1])
            else:
                print("The input image has too many dimensions for DeepImageJ.")

            if len(output_dim)==4:
                if output_dim[-1] is None:
                    self.OutputOrganization0 = 'bcyx'
                else:
                    self.OutputOrganization0 = 'byxc'
            elif len(output_dim)==5:
                if output_dim[-1] is None:
                    self.OutputOrganization0 = 'bcyxz'
                else:
                    self.OutputOrganization0 = 'byxzc'
            else:
                print("The output has too many dimensions for DeepImageJ.")
        else:
            self.FixedPatch = 'true'
            self.PatchSize = np.str(input_dim[2])
            if len(input_dim) == 4:
                if input_dim[-1] < input_dim[-2] and input_dim[-1] < input_dim[-3]:
                    self.InputOrganization0 = 'byxc'
                    self.Channels = np.str(input_dim[-1])
                else:
                    self.InputOrganization0 = 'bcyx'
                    self.Channels = np.str(input_dim[1])
            elif len(input_dim) == 5:
                if input_dim[-1] < input_dim[-2] and input_dim[-1] < input_dim[-3]:
                    self.InputOrganization0 = 'byxzc'
                    self.Channels = np.str(input_dim[-1])
                else:
                    self.InputOrganization0 = 'bcyxz'
                    self.Channels = np.str(input_dim[1])
            else:
                print("The input image has too many dimensions for DeepImageJ.")

            if len(output_dim)==4:
                if output_dim[-1] < output_dim[-2] and output_dim[-1] < output_dim[-3]:
                    self.OutputOrganization0 = 'byxc'
                else:
                    self.OutputOrganization0 = 'bcyx'
            elif len(output_dim)==5:
                if output_dim[-1] < output_dim[-2] and output_dim[-1] < output_dim[-3]:
                    self.OutputOrganization0 = 'byxzc'
                else:
                    self.OutputOrganization0 = 'bcyxz'
            else:
                print("The output has too many dimensions for DeepImageJ.")

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
        dim = np.ones(len(input_shape)-2, dtype=np.int)
        if self.FixedPatch == 'false':
            min_size = 50 * np.int(self.MinimumSize)

            if self.InputOrganization0 == 'byxc' or self.InputOrganization0 == 'byxzc':
                dim = np.concatenate(([1],min_size*dim, [input_shape[-1]]))
                null_im = np.zeros(dim, dtype=np.float32)
            else:
                dim = np.concatenate(([1, input_shape[-1]], min_size * dim))
                null_im = np.zeros(dim, dtype=np.float32)
        else:
            null_im = np.zeros((input_shape)
                               , dtype=np.float32)
            # null_im = np.expand_dims(null_im, axis=0)
            min_size = np.int(self.PatchSize)

        point_im = np.zeros_like(null_im)
        min_size = np.int(min_size / 2)

        if self.InputOrganization0 == 'byxc':
            point_im[0, min_size, min_size] = 1
        elif self.InputOrganization0 == 'byxzc':
            point_im[0, min_size, min_size, min_size] = 1
        elif self.InputOrganization0 == 'bcyx':
            point_im[0, :, min_size, min_size] = 1
        else:
            point_im[0, :, min_size, min_size, min_size] = 1

        result_unit = tf_model.predict(np.concatenate((null_im, point_im)))

        D = np.abs(result_unit[0] - result_unit[1]) > 0

        if self.OutputOrganization0 == 'byxc' or self.OutputOrganization0 == 'byxzc':
            D = D[..., 0]
        else:
            D = D[0]
        if self.OutputOrganization0 == 'byxc':
            ind = np.where(D[:min_size, :min_size] == 1)
        else:
            ind = np.where(D[:min_size, :min_size, :min_size] == 1)
        halo = np.min(ind[1])
        halo = min_size - halo + 1

        return halo

    def export_model(self, tf_model, deepimagej_model_path, **kwargs):
        """
        Main function to export the model as a bundled model of DeepImageJ
        tf_model:              tensorflow/keras model
        deepimagej_model_path: directory where DeepImageJ model is stored.
        """
        # Save the mode as protobuffer
        self.save_tensorflow_pb(tf_model, deepimagej_model_path)

        # extract the information about the testing image
        test_info = self.test_info
        io.imsave(os.path.join(deepimagej_model_path, 'exampleImage.tiff'), self.test_info.InputImage)
        io.imsave(os.path.join(deepimagej_model_path, 'resultImage.tiff'), self.test_info.OutputImage)
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
        print(
            "DeepImageJ model was successfully exported as {0}.zip. You can download and start using it in DeepImageJ.".format(
                deepimagej_model_path))

    def save_tensorflow_pb(self, tf_model, deepimagej_model_path):
        # Check whether the folder to save the DeepImageJ bundled model exists.
        # If so, it needs to be removed (TensorFlow requirements)
        # -------------- Other definitions -----------
        W = '\033[0m'  # white (normal)
        R = '\033[31m'  # red
        if os.path.exists(deepimagej_model_path):
            print(R + '!! WARNING: DeepImageJ model folder already existed and has been removed !!' + W)
            shutil.rmtree(deepimagej_model_path)

        import tensorflow as tf
        TF_VERSION = tf.__version__
        print("DeepImageJ model will be exported using TensorFlow version {0}".format(TF_VERSION))
        if TF_VERSION[:3] == "2.3":
            print(
                R + "DeepImageJ plugin is only compatible with TensorFlow version 1.x, 2.0.0, 2.1.0 and 2.2.0. Later versions are not suported in DeepImageJ." + W)

        def _save_model():
            if tf_version == 2:
                """TODO: change it once TF 2.3.0 is available in JAVA"""
                from tensorflow.compat.v1 import saved_model
                from tensorflow.compat.v1.keras.backend import get_session
            else:
                from tensorflow import saved_model
                from keras.backend import get_session

            builder = saved_model.builder.SavedModelBuilder(deepimagej_model_path)

            signature = saved_model.signature_def_utils.predict_signature_def(
                inputs={'input': tf_model.input},
                outputs={'output': tf_model.output})

            signature_def_map = {saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}

            builder.add_meta_graph_and_variables(get_session(),
                                                 [saved_model.tag_constants.SERVING],
                                                 signature_def_map=signature_def_map)
            builder.save()
            print("TensorFlow model exported to {0}".format(deepimagej_model_path))

        if TF_VERSION[0] == '1':
            tf_version = 1
            _save_model()
        else:
            tf_version = 2
            """TODO: change it once TF 2.3.0 is available in JAVA"""
            from tensorflow.keras.models import clone_model
            _weights = tf_model.get_weights(tf_model)
            with tf.Graph().as_default():
                # clone model in new graph and set weights
                _model = clone_model(tf_model)
                _model.set_weights(_weights)
                _save_model()

def write_config(Config, TestInfo, path2save):
    """
    - Config:       Class with all the information about the model's architecture and pre/post-processing
    - TestInfo:   Metadata of the image provided as an example
    - path2save:  path to the template of the configuration file.
    It can be downloaded from:
      https://raw.githubusercontent.com/deepimagej/python4deepimagej/blob/master/xml/config_template.xml
    The function updates the fields in the template provided with the
    information about the model and the example image.
    """
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/deepimagej/pydeepimagej/bioimage-yaml/pydeepimagej/yaml/bioimage.config_template.yaml",
        "bioimage.config_template.yaml")
    try:
        with open('bioimage.config_template.yaml') as file:
            YAML_dict = yaml.full_load(file)
    except:
        print("config_template.xml not found.")

    YAML_dict['name']: Config.Name
    YAML_dict['description'] = Config.Description
    YAML_dict['authors'] = Config.Authors
    CITE = {'doi': Config.DOI,
            'text': Config.References}
    YAML_dict['cite'] = CITE
    YAML_dict['documentation']: Config.Documentation
    YAML_dict['date'] = Config.Date
    YAML_dict['covers'] = 'My favourite image'
    YAML_dict['format_version'] = Config.Format_version
    YAML_dict['license'] = Config.License
    YAML_dict['framework'] = 'TensorFlow'
    YAML_dict['language'] = 'Java'
    YAML_dict['source'] = Config.Source
    YAML_dict['tags'] = Config.Tags
    YAML_dict['test_input'] = './exampleImage.tiff'
    YAML_dict['test_output'] = './resultImage.tiff'

    DIJ_CONFIG = {
        'deepimagej': {
            'model_keys': {'model_tag': 'tf.saved_model.tag_constants.SERVING',
                           'signature_definition': 'tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY'},
            'test_information': {
                'device': 'CPU',
                'input_size': '[{}]'.format(TestInfo.Input_shape),
                'output_size': '[{}]'.format(TestInfo.Output_shape),
                'pixel_size': '[{}]'.format(TestInfo.PixelSize),
                'memory_peak': 'null',
                'runtime': 'null'
            }
        }
    }
    YAML_dict['config'] = DIJ_CONFIG

    MODEL_ID = {
        'sha256': None,
        'source': './saved_model.pb'
    }
    YAML_dict['model'] = MODEL_ID

    WEIGHTS = [{'id': Config.WeightsVersion,
                'name': Config.WeightsVersion,
                'description': 'null',
                'sha256': 'null',
                'source': './variables'
                }]
    YAML_dict['weights'] = WEIGHTS
    YAML_dict['inputs'] = input_definition(Config)
    YAML_dict['outputs'] = output_definition(Config)

    YAML_dict['prediction']['weights'] = WEIGHTS[0]['id']
    YAML_dict['prediction']['dependencies'] = 'deepimagej'

    YAML_dict['prediction']['preprocess'] = {'kwargs': {'{}'.format(Config.Preprocessing[0])},
                                             'spec': 'deepimagej.runMacro::preprocessing'}
    print("Preprocessing macro '{}' set by default".format(Config.Preprocessing[0]))

    YAML_dict['prediction']['postprocess'] = {'kwargs': {'{}'.format(Config.Postprocessing[0])},
                                              'spec': 'deepimagej.runMacro::postprocessing'}
    print("Postprocessing macro '{}' set by default".format(Config.Postprocessing[0]))

    try:
        with open(os.path.join(path2save, 'config.yaml'), 'w') as file:
            documents = yaml.dump(YAML_dict, file)
            print("DeepImageJ configuration file exported.")
    except:
        print("The directory {} does not exist.".format(path2save))

def input_definition(Config):

    if Config.FixedPatch == 'true':
        shape_dict = {'exact': '[{}]'.format(Config.InputTensorDimensions)}
    else:
        min_size = np.ones(len(Config.ModelInput) - 2, dtype=np.int)
        if Config.InputOrganization0 == 'byxc' or Config.InputOrganization0 == 'byxzc':
            step_size = np.concatenate(([0], Config.MinimumSize * min_size, [0]))
            min_size = np.concatenate(([1], Config.MinimumSize * min_size, [Config.ModelInput[-1]]))
        else:
            step_size = np.concatenate(([0, 0], Config.MinimumSize * min_size))
            min_size = np.concatenate(([1, Config.ModelInput[-1]], Config.MinimumSize * min_size))

        shape_dict = {'min': '{}'.format(min_size),
                      'step': '{}'.format(step_size)}

    INPUTS = [{'name': 'raw',
               'axes': Config.InputOrganization0,
               'data_type': 'float32',
               'data_range': '[-inf, inf]',
               'shape': shape_dict}]
    return INPUTS

def output_definition(Config):
    if Config.OutputOrganization0 is not 'list':
        halo = np.ones(len(Config.ModelOutput) - 2, dtype=np.int)
        if Config.OutputOrganization0 == 'byxc' or Config.OutputOrganization0 == 'byxzc':
            halo = np.concatenate(([0], Config.Padding * halo, [0]))
        else:
            halo = np.concatenate(([0, 0], Config.Padding * min_size))

    shape_dict = {'reference_input': '{}'.format(step_size),
                  'offset': '{}'.Config.OutputOffset,
                  'scale': '{}'.Config.OutputScale}

    OUTPUTS = [{'axes': Config.OutputOrganization0,
                'data_range': None,
                'data_type': 'float32',
                'halo': '{}'.format(halo),
                'name': 'null',
                'shape': shape_dict}]
    return OUTPUTS
>>>>>>> Stashed changes
