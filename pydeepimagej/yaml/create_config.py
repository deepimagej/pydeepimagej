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
import numpy as np
import urllib
import shutil
from skimage import io
from datetime import datetime
from ..DeepImageJConfig import DeepImageJConfig
from ruamel import yaml
from ruamel.yaml import YAML
import hashlib
from zipfile import ZipFile
# import get_specification
from .bioimageio_specifications import get_specification

def FSlist(l):
    """
    Concrete list into flow-style (default is block style)
    """
    from ruamel.yaml.comments import CommentedSeq
    cs = CommentedSeq(l)
    cs.fa.set_flow_style()
    return cs


class colors:
    WHITE = '\033[0m'
    RED = '\033[31m'
    GREEN = '\033[32m'


def hash_sha256(filename):
    """
    filename: full path together with the name of the file for which the hashcode is calculated.
    """
    with open(filename, "rb") as f:
        bytes = f.read()  # read entire file as bytes
        sha256 = hashlib.sha256(bytes).hexdigest();
    return sha256


def get_dimensions(tf_model, MinimumSize):
    """
    Calculates the array organization and shapes of inputs and outputs.
    It only works for TensorFlow models
    """
    input_dim = tf_model.input_shape
    output_dim = tf_model.output_shape
    if len(output_dim) < 4:
        OutputOrganization0 = 'list'
    # Deal with the order of the dimensions and whether the size is fixed
    # or not
    if input_dim[2] is None:
        FixedPatch = 'false'
        # MinimumSize is a list with as many numbers as dimensions has the input image
        PatchSize = MinimumSize * (len(input_dim) - 1)
        if len(input_dim) == 4:
            if input_dim[-1] is None:
                InputOrganization0 = 'bcyx'
                Channels = np.str(input_dim[1])
                PatchSize = [input_dim[0]] + MinimumSize
            else:
                InputOrganization0 = 'byxc'
                Channels = np.str(input_dim[-1])
                PatchSize = MinimumSize + [input_dim[-1]]
        elif len(input_dim) == 5:
            if input_dim[-1] is None:
                InputOrganization0 = 'bcyxz'
                Channels = np.str(input_dim[1])
                PatchSize[0] = input_dim[0]
            else:
                InputOrganization0 = 'byxzc'
                Channels = np.str(input_dim[-1])
                PatchSize[-1] = input_dim[-1]
        else:
            print("The input image has too many dimensions for DeepImageJ.")

        if len(output_dim) < 3:
            # Output is a list
            OutputOrganization0 = 'null'
        elif len(output_dim) == 3:
            # This case might be completely unusual
            OutputOrganization0 = 'byx'
        elif len(output_dim) == 4:
            if output_dim[-1] is None:
                OutputOrganization0 = 'bcyx'
            else:
                OutputOrganization0 = 'byxc'
        elif len(output_dim) == 5:
            if output_dim[-1] is None:
                OutputOrganization0 = 'bcyxz'
            else:
                OutputOrganization0 = 'byxzc'
        else:
            print("The output has too many dimensions for DeepImageJ.")


    else:
        FixedPatch = 'true'
        PatchSize = input_dim[1:-1]
        if len(input_dim) == 4:
            if input_dim[-1] < input_dim[-2] and input_dim[-1] < input_dim[-3]:
                InputOrganization0 = 'byxc'
                Channels = np.str(input_dim[-1])
            else:
                InputOrganization0 = 'bcyx'
                Channels = np.str(input_dim[1])
        elif len(input_dim) == 5:
            if input_dim[-1] < input_dim[-2] and input_dim[-1] < input_dim[-3]:
                InputOrganization0 = 'byxzc'
                Channels = np.str(input_dim[-1])
            else:
                InputOrganization0 = 'bcyxz'
                Channels = np.str(input_dim[1])
        else:
            print("The input image has too many dimensions for DeepImageJ.")

        if len(output_dim) < 3:
            # Output is a list
            OutputOrganization0 = 'null'
        elif len(output_dim) == 3:
            # This case might be completely unusual
            OutputOrganization0 = 'byx'
        elif len(output_dim) == 4:
            if output_dim[-1] < output_dim[-2] and output_dim[-1] < output_dim[-3]:
                OutputOrganization0 = 'byxc'
            else:
                OutputOrganization0 = 'bcyx'
        elif len(output_dim) == 5:
            if output_dim[-1] < output_dim[-2] and output_dim[-1] < output_dim[-3]:
                OutputOrganization0 = 'byxzc'
            else:
                OutputOrganization0 = 'bcyxz'
        else:
            print("The output has too many dimensions for DeepImageJ.")
    input_dim = [1 if v is None else v for v in input_dim]
    output_dim = [1 if v is None else v for v in output_dim]

    return input_dim, output_dim, InputOrganization0, OutputOrganization0, FixedPatch, PatchSize


def _pixel_half_receptive_field(model_class, tf_model):
    """
    The halo is equivalent to the receptive field of one pixel. This value
    is used for image reconstruction when a entire image is processed.
    It only works for TensorFlow models
    """
    input_shape = tf_model.input_shape
    dim = np.ones(len(input_shape) - 2, dtype=np.int)
    if model_class.FixedPatch == 'false':
        min_size = [50 * np.int(m) for m in model_class.MinimumSize]

        if model_class.InputOrganization0 == 'byxc' or model_class.InputOrganization0 == 'byxzc':
            dim = np.concatenate(([1], min_size * dim, [input_shape[-1]]))
            null_im = np.zeros(dim, dtype=np.float32)
        else:
            dim = np.concatenate(([1, input_shape[-1]], min_size * dim))
            null_im = np.zeros(dim, dtype=np.float32)
    else:
        null_im = np.zeros((input_shape[1:]), dtype=np.float32)
        null_im = np.expand_dims(null_im, axis=0)
        min_size = model_class.PatchSize

    point_im = np.zeros_like(null_im)

    min_size = [int(m / 2) for m in min_size]

    if model_class.InputOrganization0 == 'byxc':
        point_im[0, min_size[0], min_size[1]] = 1
    elif model_class.InputOrganization0 == 'byxzc':
        point_im[0, min_size[0], min_size[1], min_size[2]] = 1
    elif model_class.InputOrganization0 == 'bcyx':
        point_im[0, :, min_size[0], min_size[1]] = 1
    else:
        point_im[0, :, min_size[0], min_size[1], min_size[2]] = 1

    result_unit = tf_model.predict(np.concatenate((null_im, point_im)))

    D = np.abs(result_unit[0] - result_unit[1]) > 0

    if model_class.OutputOrganization0 == 'byxc' or model_class.OutputOrganization0 == 'byxzc':
        D = D[..., 0]
    else:
        D = D[0]
    if model_class.OutputOrganization0 == 'byxc':
        ind = np.where(D[:min_size[0], :min_size[1]] == 1)
    else:
        ind = np.where(D[:min_size[0], :min_size[1], :min_size[2]] == 1)
    halo = np.min(ind[1])
    halo = min_size - halo + 1

    halo = [np.max((0, h)) for h in halo]

    return halo


def save_tensorflow_pb(model_class, tf_model, deepimagej_model_path):
    # Check whether the folder to save the DeepImageJ bundled model exists.
    # If so, it needs to be removed (TensorFlow requirements)

    if os.path.exists(deepimagej_model_path):
        print(colors.RED + '!! WARNING: DeepImageJ model folder already existed and has been removed !!' + colors.WHITE)
        shutil.rmtree(deepimagej_model_path)

    import tensorflow as tf
    TF_VERSION = tf.__version__
    print("DeepImageJ model will be exported using TensorFlow version {0}".format(TF_VERSION))
    if TF_VERSION[:3] == "2.3":
        print(
            colors.RED + "DeepImageJ plugin is only compatible with TensorFlow version 1.x, 2.0.0, 2.1.0 and 2.2.0. Later versions are not suported in DeepImageJ." + colors.WHITE)

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

        ziped_model = os.path.join(deepimagej_model_path, model_class.WeightsSource)

        filePaths = []

        # Add multiple files to the zip
        # zipObj.write(os.path.join(deepimagej_model_path, 'saved_model.pb'), os.path.basename(os.path.join(deepimagej_model_path, 'saved_model.pb')))
        for folderNames, subfolder, filenames in os.walk(os.path.join(deepimagej_model_path)):
            for filename in filenames:
                # create complete filepath of file in directory
                filePaths.append(os.path.join(folderNames, filename))

        zipObj = ZipFile(ziped_model, 'w')
        for f in filePaths:
            # Add file to zip
            zipObj.write(f, os.path.relpath(f, deepimagej_model_path))
        # close the Zip File
        zipObj.close()

        try:
            shutil.rmtree(os.path.join(deepimagej_model_path, 'variables'))
            os.remove(os.path.join(deepimagej_model_path, 'saved_model.pb'))
        except:
            print("TensorFlow bundled model was not removed after compression")

        with open(ziped_model, "rb") as f:
            bytes = f.read()  # read entire file as bytes
            readable_hash = hashlib.sha256(bytes).hexdigest();
        print("TensorFlow model exported to {0}".format(deepimagej_model_path))

        return readable_hash

    if TF_VERSION[0] == '1':
        tf_version = 1
        ModelHash = _save_model()
    else:
        tf_version = 2
        """TODO: change it once TF 2.3.0 is available in JAVA"""
        from tensorflow.keras.models import clone_model
        _weights = tf_model.get_weights()
        with tf.Graph().as_default():
            # clone model in new graph and set weights
            _model = clone_model(tf_model)
            _model.set_weights(_weights)
            ModelHash = _save_model()
    return ModelHash


def weights_definition(Config, YAML_dict):
    YAML_dict['weights'] = {}

    for W in Config.Weights:
        # TODO: Consider multiple outputs and inputs
        WEIGHTS = {'source': './' + W.WeightsSource,
                   'sha256': W.ModelHash
                   }
        if hasattr(W, 'FormatParent'):
            WEIGHTS.update({'parent': W.FormatParent})

        if hasattr(W, 'Authors'):
            WEIGHTS.update({'authors': W.Authors})

        if W.Framework == 'TensorFlow':
            YAML_dict['weights'].update({'tensorflow_saved_model_bundle': WEIGHTS})

        elif W.Framework == 'KerasHDF5':
            YAML_dict['weights'].update({'keras_hdf5': WEIGHTS})

        elif W.Framework == 'TensoFlow-JS':
            YAML_dict['weights'].update({'tensorflow_js': WEIGHTS})

        elif W.Framework == 'PyTorch-JS':
            YAML_dict['weights'].update({'pytorch_script': WEIGHTS})

    return YAML_dict


def input_definition(Config, YAML_dict):
    # TODO: Consider multiple outputs and inputs
    input_data_range = "[-inf, inf]"
    input_dict = {'name': 'input',
                  'axes': Config.InputOrganization0,
                  'data_type': 'float32',
                  'data_range': input_data_range}
    if Config.BioImage_Preprocessing is not None:
        input_dict['preprocessing'] = Config.BioImage_Preprocessing

    YAML_dict['inputs'] = [input_dict]

    if Config.FixedPatch == 'true':
        YAML_dict['inputs'][0]['shape'] = FSlist(Config.InputTensorDimensions)
    else:
        # min_size = np.ones(len(Config.ModelInput) - 2, dtype=np.int)
        if Config.InputOrganization0 == 'byxc' or Config.InputOrganization0 == 'byxzc':
            step_size = [0] + Config.MinimumSize + [0]
            min_size = [1] + Config.MinimumSize + [Config.ModelInput[-1]]
        else:
            step_size = [0, 0] + Config.MinimumSize
            min_size = [1, Config.ModelInput[-1]] + Config.MinimumSize
        YAML_dict['inputs'][0]['shape'] = {'min': FSlist(min_size),
                                           'step': FSlist(step_size)}
    return YAML_dict


def output_definition(Config, YAML_dict):
    # TODO: Consider multiple outputs and inputs
    output_data_range = "[-inf, inf]"
    output_dict = {'name': 'output',
                   'axes': Config.OutputOrganization0,
                   'data_range': output_data_range,
                   'data_type': 'float32'}
    if Config.BioImage_Postprocessing is not None:
        output_dict['postprocessing']= Config.BioImage_Postprocessing

    if Config.OutputOrganization0 != 'list' and Config.OutputOrganization0 != 'null':
        # TODO: consider 3D+ outputs for the halo
        if Config.OutputOrganization0[-1] == 'c':
            halo = list([0] + Config.Halo + [0])
        else:
            halo = list([0, 0] + Config.Halo)
        halo = [int(h) for h in halo]
    else:
        # Note that the output has not the dimensions of the input so this might not be conceptually correct.
        halo = [0 for v in Config.ModelInput]

    YAML_dict['outputs'] = [output_dict]
    YAML_dict['outputs'][0]['halo'] = FSlist(halo)
    YAML_dict['outputs'][0]['shape'] = {'reference_input': 'input',
                                        'offset': FSlist(Config.OutputOffset),
                                        'scale': FSlist(Config.OutputScale)}
    return YAML_dict


def write_config(Config, path2save):
    """
    - Config:       Class with all the information about the model's architecture and pre/post-processing
    - TestInfo:   Metadata of the image provided as an example
    - path2save:  path to the template of the configuration file.
    It can be downloaded from:
      https://raw.githubusercontent.com/deepimagej/pydeepimagej/master/pydeepimagej/yaml/bioimage.io.config_template.yaml
    The function updates the fields in the template provided with the
    information about the model and the example image.
    """
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/deepimagej/pydeepimagej/master/pydeepimagej/yaml/bioimage.io.config_template.yaml",
        "bioimage.io.config_template.yaml")
    try:
        yaml = YAML()
        with open('bioimage.io.config_template.yaml') as f:
            YAML_dict = yaml.load(f)
    except:
        print("bioimage.io.config_template.yaml not found.")

    YAML_dict['name'] = Config.Name
    YAML_dict['description'] = Config.Description
    YAML_dict['authors'] = Config.Authors
    if Config.References is not None and Config.References != 'null':
        if len(Config.References) == len(Config.DOI):
            YAML_dict['cite'] = [{'doi': Config.DOI[i],
                                  'text': Config.References[i]} for i in range(len(Config.References))]
        else:
            YAML_dict['cite'] = [{'doi': None,
                                  'text': Config.References[i]} for i in range(len(Config.References))]
    else:
        YAML_dict['cite'] = None

    if hasattr(Config, 'Parent') and Config.Parent is not None:
        YAML_dict['parent'] = {'uri': Config.Parent,
                               'sha256': None
                               }

    YAML_dict['documentation'] = 'null' if Config.Documentation is None else Config.Documentation
    YAML_dict['timestamp'] = Config.Timestamp
    YAML_dict['covers'] = 'null' if Config.Covers is None else Config.Covers
    YAML_dict['format_version'] = Config.Format_version
    YAML_dict['license'] = 'null' if Config.License is None else Config.License
    YAML_dict['framework'] = 'null' if Config.Framework is None else Config.Framework
    YAML_dict['language'] = 'Java'
    YAML_dict['source'] = 'null' if Config.Source is None else Config.Source
    YAML_dict['tags'] = Config.Tags
    YAML_dict['git_repo'] = 'null' if Config.GitHub is None else Config.GitHub
    YAML_dict['packaged_by'] = 'null' if Config.PackagedBy is None else Config.PackagedBy
    YAML_dict['attachments'] = 'null' if Config.Attachments is None else Config.Attachments
    YAML_dict['parent'] = 'null' if Config.Parent is None else Config.Parent

    if hasattr(Config, 'test_info'):
        YAML_dict['sample_inputs'] = ['./exampleImage.npy']
        YAML_dict['test_inputs'] = ['./exampleImage.tif']

        if Config.test_info.Output_type == 'image':
            YAML_dict['sample_outputs'] = ['.resultImage.npy']
            YAML_dict['test_outputs'] = ['.resultImage.tif']
        else:
            YAML_dict['sample_outputs'] = ['.resultTable.npy']
            YAML_dict['test_outputs'] = ['.resultTable.csv']

    YAML_dict = weights_definition(Config, YAML_dict)
    YAML_dict = input_definition(Config, YAML_dict)
    YAML_dict = output_definition(Config, YAML_dict)

    dij_config = bioimageio_spec_config_deepimagej(Config, YAML_dict)
    YAML_dict['config'] = dij_config

    YAML_dict.default_flow_style = False

    try:
        yaml = YAML()
        yaml.default_flow_style = False
        with open(os.path.join(path2save, 'model.yaml'), 'w', encoding='UTF-8') as f:
            yaml.dump(YAML_dict, f)
            print("DeepImageJ configuration file exported.")
    except:
        print(colors.RED + 'The specification file model.yaml could not be created' + colors.WHITE)


def bioimageio_spec_config_deepimagej(Config, YAML_dict):
    if Config.Preprocessing is not None:
        preprocess = [{'spec': 'ij.IJ::runMacroFile', 'kwargs': '{}'.format(step)} for step in Config.Preprocessing]
    else:
        preprocess = None
    if Config.Postprocessing is not None:
        postprocess = [{'spec': 'ij.IJ::runMacroFile', 'kwargs': '{}'.format(step)} for step in Config.Postprocessing]
    else:
        postprocess = None
    if hasattr(Config, 'test_info'):
        if Config.test_info.PixelSize is not None:
            if len(Config.test_info.PixelSize) == 3:
                pixel_size = {'x': '{} µm'.format(Config.test_info.PixelSize[0]),
                              'y': '{} µm'.format(Config.test_info.PixelSize[1]),
                              'z': '{} µm'.format(Config.test_info.PixelSize[2])}
            else:
                pixel_size = {'x': '{} µm'.format(Config.test_info.PixelSize[0]),
                              'y': '{} µm'.format(Config.test_info.PixelSize[1]),
                              'z': '1.0 pixel'}
        else:
            pixel_size = {'x': '1.0 pixel',
                          'y': '1.0 pixel',
                          'z': '1.0 pixel'}

        test_information = {
            'device': None,  # TODO: check if DeepImageJ admits null
            'inputs': {
                'name': 'input',
                'size': Config.test_info.Input_shape,
                'pixel_size': pixel_size
            },
            'outputs': {
                'name': 'output',
                'type': Config.test_info.Output_type,
                'size': Config.test_info.Output_shape
            },
            'memory_peak': Config.test_info.MemoryPeak,
            'runtime': Config.test_info.Runtime
        }
    else:
        test_information = YAML_dict['config']['deepimagej']['test_information']

    dij_config = {
        'deepimagej': {
            'pyramidal_model': Config.pyramidal_model,
            'allow_tiling': Config.allow_tiling,
            'model_keys': {'tensorflow_model_tag': 'tf.saved_model.tag_constants.SERVING',
                           'tensorflow_siganture_def': 'tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY'},
            'test_information': test_information,
            'prediction': {
                'preprocess': preprocess,
                'postprocess': postprocess
            }
        }
    }
    return dij_config


class BioImageModelZooConfig(DeepImageJConfig):
    def __init__(self, tf_model, MinimumSize):
        # Import all the information needed for DeepImageJ
        DeepImageJConfig.__init__(self, tf_model)
        # New fields for the bioimage.io configuration file
        self.Timestamp = datetime.now()
        self.MinimumSize = MinimumSize
        self.Description = None
        self.DOI = None
        self.Documentation = None
        self.Format_version = '0.3.0'  # bioimage.io
        self.License = 'BSD-3'
        self.Tags = ['deepimagej']
        self.Covers = None
        # TODO: detect model framework (at least among pytorch and TF)?
        self.Framework = 'TensorFlow'
        self.GitHub = None
        self.Source = None
        self.PackagedBy = ['pydeepimagej']
        self.Attachments = None
        self.Parent = None
        # self.WeightsTorchScript = 'pytorch_script.pt'
        try:
            I, O, IA, OA, F, P = get_dimensions(tf_model, self.MinimumSize)
            self.InputTensorDimensions = I
            self.OutputTensorDimensions = O
            self.InputOrganization0 = IA
            self.OutputOrganization0 = OA
            self.FixedPatch = F
            self.PatchSize = P
        except:
            print(colors.GREEN + 'pydeepimagej is not able to specify the inputs and output information.')
            print('Please, include the parameters (InputTensorDimensions, OutputTensorDimensions,')
            print(
                'InputOrganization0, OutputOrganization0, FixedPatch, PatchSize and Padding,  manually.' + colors.WHITE)
        try:
            # Receptive field of the network to process input
            if self.OutputOrganization0 != 'list' and self.OutputOrganization0 != 'null':
                self.Halo = _pixel_half_receptive_field(self, tf_model)
        except:
            print(colors.GREEN + 'The halo of the model is undetermined and will be set as zero' + colors.WHITE)
            # batch and channel axis are not considered
            self.Halo = np.zeros(len(tf_model.input_shape) - 2, dtype=np.int)

        self.ModelInput = tf_model.input_shape
        self.ModelOutput = tf_model.output_shape
        self.OutputOffset = [0 for v in self.ModelInput]
        self.OutputScale = [1 for v in self.ModelInput]
        self.pyramidal_model = False
        if self.OutputOrganization0 == 'list' or self.OutputOrganization0 == 'null':
            self.allow_tiling = False
        else:
            self.allow_tiling = True
        self.Preprocessing = None
        self.Preprocessing_files = None
        self.Postprocessing = None
        self.Postprocessing_files = None
        self.BioImage_Preprocessing = None
        self.BioImage_Postprocessing = None

    class TestImage:
        def __add__(self, input_im, output_im, output_type, pixel_size):
            """
            pixel size is a float type vector with the size for each dimension given in microns
            """
            self.Input_shape = ' x '.join([np.str(i) for i in input_im.shape])
            self.InputImage = input_im
            self.Output_shape = ' x '.join([np.str(i) for i in output_im.shape])
            self.Output_type = output_type
            self.OutputImage = output_im
            self.MemoryPeak = None
            self.Runtime = None
            self.PixelSize = pixel_size

    def create_covers(self, im_list):
        """
        - im_list: list of images that will be used for the covers.
        Images are assumed to have dimension order (Z/time/channels, height, width, ...).
        In case the image has more than 2 dimensions, the first 2D image is chosen.

        The images are stored as png and to visualize them online, their values
        are interpolated to the [0,255] range and converted into uint8
        """
        self.CoverImages = []
        for im in im_list:
            while len(im.shape) > 2:
                im = im[0]
            im = np.interp(im, (im.min(), im.max()), (0, 255))
            im = im.astype(np.uint8)
            self.CoverImages.append(im)

    def add_test_info(self, input_im, output_im, pixel_size=None):
        self.test_info = self.TestImage()
        if self.OutputOrganization0 == 'list' or self.OutputOrganization0 == 'null':
            output_type = 'ResultsTable'
        else:
            output_type = 'image'
        self.test_info.__add__(input_im, output_im, output_type, pixel_size)

    def add_bioimageio_spec(self, processing, name, **kwargs):

        specs = get_specification(name, **kwargs)

        if processing == 'pre-processing':

            if self.BioImage_Preprocessing is not None:
                self.BioImage_Preprocessing.append(specs)
            else:
                self.BioImage_Preprocessing = [specs]

        elif processing == 'post-processing':
          
            if self.BioImage_Postprocessing is not None:
                self.BioImage_Postprocessing.append(specs)
            else:
                self.BioImage_Postprocessing = [specs]
        else:

          print("add_bioimage_spec only accepts 'pre-processing' or 'post_processing' input process name.")

    class WeightsFormat:
        def __init__(self, model, format, parent, authors):
            if parent is not None:
                self.FormatParent = parent
            if authors is not None:
                self.Authors = authors
            self.Framework = format
            self.Model = model

    def add_weights_formats(self, model, format, parent=None, authors=None):
        if not hasattr(self, 'Weights'):
            self.Weights = []

        self.Weights.append(self.WeightsFormat(model, format, parent, authors))

    def export_model(self, deepimagej_model_path, **kwargs):
        """
        Main function to export the model as a bundled model of DeepImageJ
        self.Weights should contain at least one model
            W = self.Weights[0]
            W.Model is a model.
        deepimagej_model_path: directory where DeepImageJ model is stored.
        """
        # # Save the mode as protobuffer
        ## TODO: Sotore JS and PyTorch models.
        for W in self.Weights:
            if W.Framework == 'TensorFlow':
                W.WeightsSource = 'tensorflow_saved_model_bundle.zip'
                W.ModelHash = save_tensorflow_pb(W, W.Model, deepimagej_model_path)

            elif W.Framework == 'KerasHDF5':
                W.WeightsSource = 'keras_model.h5'
                W.Model.save(os.path.join(deepimagej_model_path, W.WeightsSource))
                W.ModelHash = hash_sha256(os.path.join(deepimagej_model_path, W.WeightsSource))

            elif W.Framework == 'TensoFlow-JS':
                W.WeightsSource = 'tensorflow_javascript.zip'
                W.ModelHash = hash_sha256(os.path.join(deepimagej_model_path, W.WeightsSource))

            elif W.Framework == 'PyTorch-JS':
                W.WeightsSource = 'pytorch_script.pt'
                W.ModelHash = hash_sha256(os.path.join(deepimagej_model_path, W.WeightsSource))

        # record which files should be attached for the BioImage Model Zoo packager
        attachments_files = []

        if hasattr(self, 'test_info'):
            # extract the information about the testing image
            io.imsave(os.path.join(deepimagej_model_path, 'exampleImage.tif'),
                      self.test_info.InputImage)
            # store numpy arrays for future bioimage.io CI
            np.save(os.path.join(deepimagej_model_path, 'exampleImage.npy'),
                    self.test_info.InputImage)

            attachments_files.append("./exampleImage.tif")

            if self.test_info.Output_type == 'image':
                io.imsave(os.path.join(deepimagej_model_path, 'resultImage.tif'),
                          self.test_info.OutputImage)
                # store numpy arrays for future bioimage.io CI
                np.save(os.path.join(deepimagej_model_path, 'resultImage.npy'),
                        self.test_info.OutputImage)

                attachments_files.append("./resultImage.tif")

            else:
                columns = ['C{}'.format(c + 1) for c in range(self.test_info.OutputImage.shape[-1])]
                columns = ','.join(columns)
                np.savetxt(os.path.join(deepimagej_model_path, 'resultTable.csv'),
                           self.test_info.OutputImage, delimiter=",",
                           header=columns, comments="")
                # store numpy arrays for future bioimage.io CI
                np.save(os.path.join(deepimagej_model_path, 'resultTable.npy'),
                        self.test_info.OutputImage)

                attachments_files.append("./resultTable.csv")

            print("Example images stored.")
        if hasattr(self, 'CoverImages'):
            # extract the information about the testing image
            for c in range(len(self.CoverImages)):
                io.imsave(os.path.join(deepimagej_model_path, self.Covers[c]),
                          self.CoverImages[c])
            print("Covers stored.")

        # Add preprocessing and postprocessing macros.
        # More than one is available, but the first one is set by default.

        if self.Preprocessing is not None:
            for i in range(len(self.Preprocessing)):
                shutil.copy2(self.Preprocessing_files[i], os.path.join(deepimagej_model_path, self.Preprocessing[i]))
                print("ImageJ macro {} included in the bundled model.".format(self.Preprocessing[i]))

                attachments_files.append("./{}".format(self.Preprocessing[i]))

        if self.Postprocessing is not None:
            for i in range(len(self.Postprocessing)):
                shutil.copy2(self.Postprocessing_files[i], os.path.join(deepimagej_model_path, self.Postprocessing[i]))
                print("ImageJ macro {} included in the bundled model.".format(self.Postprocessing[i]))

                attachments_files.append("./{}".format(self.Postprocessing[i]))


        # Update attachments
        self.Attachments = {'files':  FSlist(attachments_files)}

        # write the DeepImageJ configuration model.yaml file according to BioImage Model Zoo
        write_config(self, deepimagej_model_path)


        # Zip the bundled model to download
        shutil.make_archive(deepimagej_model_path, 'zip', deepimagej_model_path)
        print(
            colors.GREEN + 'DeepImageJ model was successfully exported as {0}.zip. You can download and start using it in DeepImageJ.'.format(
                deepimagej_model_path) + colors.WHITE)
