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
# scale_range normalize the tensor with percentile normalization
    # kwargs
    # mode can be one of per_sample (percentiles are computed for each sample individually), per_dataset (percentiles are computed for the entire dataset). For a fixed scaling use scale linear.
    # axes the subset of axes to normalize jointly. For example xy to normalize the two image axes for 2d data jointly. The batch axis (b) is not valid here.
    # min_percentile the lower percentile used for normalization, in range 0 to 100. Default value: 0.
    # max_percentile the upper percentile used for normalization, in range 1 to 100. Has to be bigger than upper_percentile. Default value: 100. The range is 1 to 100 instead of 0 to 100 to avoid mistakenly accepting percentiles specified in the range 0.0 to 1.0
    # reference_implementaion
# binarize
    # kwargs
    # threshold the fixed threshold
    # reference_implemation
# clip clip the tensor
    # kwargs
    # min minimum value for clipping
    # max maximum value for clipping
    # reference_implementation
# scale_linear
    # kwargs
    # gain multiplicative factor
    # offset additive factor
    # axes the subset of axes to scale jointly. For example xy to scale the two image axes for 2d data jointly. The batch axis (b) is not valid here.
    # reference_implementation
# sigmoid .
    # kwargs None
    # reference_implementation
# zero_mean_unit_variance
    # kwargs
    # mode can be one of fixed (fixed values for mean and variance), per_sample (mean and variance are computed for each sample individually), per_dataset (mean and variance are computed for the entire dataset)
    # axes the subset of axes to normalize jointly. For example xy to normalize the two image axes for 2d data jointly. The batch axis (b) is not valid here.
    # mean the mean value(s) to use for mode == fixed. For example [1.1, 2.2, 3.3] in the case of a 3 channel image where the channels are not normalized jointly.
    # std the standard deviation values to use for mode == fixed. Analogous to mean.
    # [eps] epsilon for numeric stability: out = (tensor - mean) / (std + eps). Default value: 10^-7.
    # reference_implementation

def scale_range(mode='per_sample',axes='xyzc', min_percentile=0, max_percentile=1):
    """
    Normalize the tensor with percentile normalization
    """
    dict_scale_range = {'scale_range': {
        'kwargs': {
            'mode': mode,
            'axes': axes,
            'min_percentile': min_percentile,
            'max_percentile': max_percentile

            }}
        }
    return dict_scale_range

def clip(threshold=0.5):
    """
    clip the tensor
    """
    dict_binarize = {'scale_range': {
        'kwargs': {
            'threshold': threshold
        }}
    }
    return dict_binarize

def binarize(threshold=0.5):
    """
    binarize the tensor with a fixed threshold, values above the threshold will be set to one, values below the threshold to zero
    """
    dict_binarize = {'binarize': {
        'kwargs': {
            'threshold': threshold
        }}
    }
    return dict_binarize

def scale_linear(gain=1, offset=0, axes='yx'):
    """
    scale the tensor with a fixed multiplicative and additive factor
    """
    dict_scale_linear = {'scale_range': {
        'kwargs': {
            'gain': gain,
            'offset': offset,
            'axes': axes
        }}
    }
    return dict_scale_linear

def sigmoid():
    """
    apply a sigmoid to the tensor
    """
    dict_sigmoid = {'sigmoid'}
    return dict_sigmoid

def zero_mean_unit_variance(mode='per_sample', axes='xyzc', mean=0, std=1, eps=1e-07):
    """
    normalize the tensor to have zero mean and unit variance
    """
    dict_zero_mean_unit_variance = {'zero_mean_unit_variance': {
        'kwargs': {
            'mode': mode,
            'axes': axes,
            'mean': mean,
            'std': std,
            'eps': eps
        }}
    }
    return dict_zero_mean_unit_variance

def get_specification(process_name, **kwargs):
    '''
    selects the corresponding specification for the processing chosen
    '''
    if process_name == 'scale_range' or process_name == 'percentile':
        return scale_range(**kwargs)
    elif process_name == 'clip':
        return clip(**kwargs)
    elif process_name == 'binarize':
        return binarize(**kwargs)
    elif process_name == 'scale_linear':
        return scale_linear(**kwargs)
    elif process_name == 'sigmoid':
        return sigmoid()
    elif process_name == 'zero_mean_unit_variance':
        return zero_mean_unit_variance(**kwargs)
    else:
        print('the process {} does not exist in the specifications of the Bioimae Model Zoo.'.format(process_name))
        return None