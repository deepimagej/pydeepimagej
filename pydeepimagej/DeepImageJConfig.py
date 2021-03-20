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

class DeepImageJConfig:
    def __init__(self, tf_model):
        # ModelInformation
        self.Name       = 'null'
        self.Authors    = 'null'
        self.URL        = 'null'
        self.Credits    = 'null'
        self.Version    = 'null'
        self.References = 'null'
        self.Date       = time.ctime()
        # Same value as 2**pooling_steps 
        # (related to encoder-decoder archtiectures) when the input size is not
        # fixed
        self.MinimumSize = '8'
        self.Preprocessing = list()
        self.Postprocessing = list()
        self.Preprocessing_files = list()
        self.Postprocessing_files = list()
    
    class TestImage:
        def __add__(self, input_im, output_im, pixel_size):
            """
            pixel size must be given in microns
            """
            self.Input_shape = '{0}x{1}'.format(input_im.shape[0], input_im.shape[1])
            self.InputImage = input_im
            self.Output_shape = '{0}x{1}'.format(output_im.shape[0], output_im.shape[1])
            self.OutputImage = output_im
            self.MemoryPeak = 'null'
            self.Runtime = 'null'
            self.PixelSize = '{0}µmx{1}µm'.format(pixel_size, pixel_size)

    def add_test_info(self, input_im, output_im, pixel_size):
        self.test_info = self.TestImage()
        self.test_info.__add__(input_im, output_im, pixel_size)

    def add_preprocessing(self, file, name):
        file_extension = file.split('.')[-1]
        name = name + '.' + file_extension
        if name.startswith('preprocessing'):
            self.Preprocessing.insert(len(self.Preprocessing),name)
        else:
            name = "preprocessing_"+name
            self.Preprocessing.insert(len(self.Preprocessing),name)
        self.Preprocessing_files.insert(len(self.Preprocessing_files), file)

    def add_postprocessing(self, file, name):
        file_extension = file.split('.')[-1]
        name = name + '.' + file_extension
        if name.startswith('postprocessing'):
            self.Postprocessing.insert(len(self.Postprocessing), name)
        else:
            name = "postprocessing_" + name
            self.Postprocessing.insert(len(self.Postprocessing), name)
        self.Postprocessing_files.insert(len(self.Postprocessing_files), file)