from __future__ import annotations

import torch
import random
import numpy as np
from math import ceil, floor

from torch.nn import AvgPool2d, BatchNorm2d, Conv2d, Dropout, Dropout2d, Flatten, LeakyReLU, Linear, MaxPool2d, PReLU, ReLU, Sequential


def _instance_from_name(class_name, *args, **kwargs):
    """
    Instantiate class by name
    """
    klass = globals()[class_name]
    class_instance = klass(*args, **kwargs)
    return class_instance



###################### MODULE TEMPLATES ########################
class ModuleTemplate_():
    '''
        Base Class for Module Template.
        Serves as a template for future Pytorch Module instance. Holds only the necessary config of the 
        future module and doesn't hold weights that's why much lighter than Pytorch module. Could be then used a part of 
        Model template - architechture of future Model.
        
        Upon __init__(self, in_shape=None, **kwargs) 'in_shape' is always present, **kwargs
        are specific to each module and saved as self.params dict (Parameter class). Also for each config we have to set up
        a range of acceptable values for further optimisation or set '.learnable = False' (self._init_param_values_list())
        When optimising config we can generate a new random set of config from corresponding range (self.gen_rand_config)

        Important! There is no direct correspondance between ModuleTemplate_ Class config and PyTorch Module
        Class config.

        After initialisation (__init__) self.in_shape attribute is determined and we can calculate self.out_shape
        (self._update_out_shape() method) for linkage with other templates when it will be used as a part of Model.

        Finally in order to get Pytorch module instance from template we need first create arguments for Pytorch Module, as not all
        Module Template args corresponds to Pytorch module args. (self._create_args(): self.config -> self._args, self._kwargs)
        Template could be then instantiated into pytorch Module and used for training ( self.instantiate_module() ).

        Example N1:
            # create simple 1 layer model from templates
            # with predetermined config
            # Conv - Flatten - Linear
            # create Conv2d template with CIFAR10 shape (3, 32, 32)
            conv_tmpl = Conv2dTmpl(in_shape = (3, 32, 32),
                                    kernel_size = 3,
                                    stride = 1,
                                    padding = True)

            # calculate the output shape
            conv_tmpl._update_out_shape()

            # get pytorch module instance
            conv = conv_tmpl.instantiate_module()

            # repeat for Linear and Flatten tmpl
            flatten_tmpl = FlattenTmpl(in_shape = conv_tmpl.out_shape)
            flatten_tmpl._update_out_shape()
            flatten = flatten_tmpl.instantiate_module()

            lin_tmpl = linTmpl(in_shape = flatten_tmpl.out_shape, out_shape = 10)
            lin_tmpl._update_out_shape()
            lin = lin_tmpl.instantiate_module()

            # create model
            model = Sequential(conv, flatten, lin)

        Example N2:

            # generate 10 models with random config using self.gen_rand_config():

            # create empty templates only with starting shape defined
            conv_tmpl = Conv2dTmpl(in_shape = (3, 32, 32))
            relu_tmpl = ReLUTmpl()
            flatten_tmpl = FlattenTmpl()
            lin_tmpl = linTmpl()

            templates = [conv_tmpl, relu_tmpl, flatten_tmpl, lin_tmpl, relu_tmpl, lin_tmpl]

            # final models
            models = []

            for i in range(10):
                next_in_shape = conv_tmpl.in_shape
                for tmpl in templates:
                    tmpl.in_shape = next_in_shape

                    # generate random config for each tmpl
                    tmpl.gen_rand_config()

                    tmpl._update_out_shape()
                    next_in_shape = tmpl.out_shape

                # set out_shape 10 for the last tmpl
                templates[-1].set_out_shape(10)

                # list of pytoch modules in model
                mods =[]
                # instantiate pytorch modules
                for tmpl in templates:
                    mods.append(tmpl.instantiate_module())

                # create pytorch model
                models.append(Sequential(*mods))
    '''

    config = {}

    # pytorch module name
    torch_name = None

    def __init__(self, in_shape=None, **kwargs):
        '''
            Creates a new ModuleTemplate instance. Saves in_shape and generates config.

            Required arguments:
            - in_shape (tuple, int or None):  shape of the input without batch size, ex: (3, 32, 32)

            Optional arguments:
            - kwargs (dict): specific config for each module as kwargs, ex: kernel_size = (3, 3)
        '''
        self.type = type(self).__name__
        self._in_shape = in_shape
        self.params = {}
        for k, v in self.config.items():
            new_val = kwargs.get(k)
            self.params[k] = v['default'] if new_val is None else new_val

        self.update_shape()
        # utils.print_nonprivate_properties(self)


    @property
    def in_shape(self):
        return self._in_shape

    @in_shape.setter
    def in_shape(self, in_shape):
        self._in_shape = in_shape
        self.update_shape()

    def update_shape(self):
        """
        Handles change of out_shape
        """
        if self.in_shape is None:
            self.out_shape = None
            self.is_zero_shape = False
        else:
            self._update_out_shape()
            self._is_zero_shape()

    def _update_out_shape(self):
        """
        Updates out_shape value when in_shape or config changes
        """
        # children overwrite this
        self.out_shape = self._in_shape

    def _is_zero_shape(self):
        """
        Zero shape check for output shape
        Sets self.is_zero_shape 'True' if at least one dimension <=0
        """
        self.is_zero_shape = False
        # multidim shape
        if type(self.out_shape) is tuple:
            for dim in self.out_shape:
                if dim < 1:
                    self.is_zero_shape = True
                    return True
        # scalar shape
        elif self.out_shape < 1:
            self.is_zero_shape = True
            return True

        return False

    def set_out_shape(self, out_shape):
        """
        We can set out_shape only for Linear type templates.
        """
        # Linear templates overwrites this
        raise Exception('Cannot set out_shape for Non linear templates (%s)' % self.torch_name)

    def gen_rand_params(self):
        # self.params = ConfigGenerator(self).get_random()
        for k, v in self.config.items():
            learning_range = v.get('range')
            if learning_range:
                self.params[k] = random.choice(learning_range)
            else:
                self.params[k] = v.get('default')
        self.update_shape()

    def get_learnable_params(self):
        """
        Returns a list of learnable params
        """
        res = {}
        for k, v in self.config.items():
            if v.get('range') is not None:
                res[k] = self.params[k]
        return res

    def instantiate_module(self):
        """
        Returns an instance of PyTorch module from template.

        self.in_shape != None

        Output:
        - instance: PyTorch module instance
        """
        # in_type = type(self.in_shape)
        # if (in_type is int) or (in_type is tuple):
        self._create_torch_args()
        return _instance_from_name(self.torch_name, *self._args, **self._kwargs)

    def _create_torch_args(self):
        '''
        Children overwrite this.

        Creates arguments from self.params for pytorch Module class
        Because sometimes there is no direct correspond between Template args
        and Pytorch Module args.

        Example for pytorch Linear(in_shape, features, bias = True):
            self._args = [self.in_shape, self.out_shape]
            self._kwargs = {'bias' : self.params['bias']}

        '''
        self._args = []
        self._kwargs = {}

    def __repr__(self):
        # tmpl_str = ', '.join((repr(self.in_shape), repr(self.params)))
        repr_str = '{:12} {:14}'.format(str(self.torch_name), str(self.out_shape))
        return repr_str

    def __eq__(self, other):
        if self.__class__.__name__ != other.__class__.__name__:
            return False

        if self.params != other.params:
            return False

        return True


class LinearTmpl(ModuleTemplate_):
    config = {
        'out_features': {
            'default': 256,
            'range': [32, 64, 128, 256, 512, 1024, 2048, 4096]
        },
        'bias': {
            'default': True
        }
    }

    def __init__(self, in_shape=None, **kwargs):
        self.torch_name = 'Linear'
        super().__init__(in_shape=in_shape, **kwargs)
        # utils.print_nonprivate_properties(self)

    def _update_out_shape(self):
        self.out_shape = self.params['out_features']

    def set_out_shape(self, out_shape):
        self.out_shape = out_shape
        self.params['out_features'] = out_shape

    def calc_maccs(self):
        self.maccs = self.in_shape * self.out_shape

    def _calc_flops(self):
        bias = self.params['bias']
        if bias:
            self.flops = 2 * self.in_shape * self.out_shape
        else:
            self.flops = (2 * self.in_shape - 1) * self.out_shape

    def _create_torch_args(self):
        self._args = [self.in_shape, self.params['out_features']]
        self._kwargs = {'bias': self.params['bias']}


class BatchNorm2dTmpl(ModuleTemplate_):
    config = {
        'eps': {
            'default': 1e-5,
        },
        'momentum': {
            'default': 0.1,
        },
        'affine': {
            'default': True,
        },
        'track_running_stats': {
            'default': True,
        }
    }

    def __init__(self, in_shape=None, **kwargs):
        self.torch_name = 'BatchNorm2d'
        super().__init__(in_shape = in_shape, **kwargs)
        # utils.print_nonprivate_properties(self)

    def _create_torch_args(self):
        self._args = [self.in_shape[0]]
        self._kwargs = {
            'eps' : self.params['eps'],
            'momentum' : self.params['momentum'],
            'affine' : self.params['affine'],
            'track_running_stats' : self.params['track_running_stats']
        }


class ReLUTmpl(ModuleTemplate_):
    config = {
        'inplace': {
            'default': False,
        }
    }

    def __init__(self, in_shape=None, **kwargs):
        self.torch_name = 'ReLU'
        super().__init__(in_shape = in_shape, **kwargs)

    def _create_torch_args(self):
        self._args = []
        self._kwargs = {
            'inplace' : self.params['inplace']
        }


class LeakyReLUTmpl(ModuleTemplate_):
    config = {
        'negative_slope': {
            'default': 0.1,
            'range': [0.1, 0.01, 0.001]
        },
        'inplace': {
            'default': False,
        }
    }

    def __init__(self, in_shape=None, **kwargs):
        self.torch_name = 'LeakyReLU'
        super().__init__(in_shape = in_shape, **kwargs)

    def _create_torch_args(self):
        self._args = []
        self._kwargs = {
            'negative_slope': self.params['negative_slope'],
            'inplace': self.params['inplace']
        }

    def __repr__(self):
        negative_slope = self.params['negative_slope']
        repr_str = '{:12} {:14} {:8}'.format( self.torch_name,
                                        str(self.out_shape),
                                        '(' + str(negative_slope) + ')' )
        return repr_str


class PReLUTmpl(ModuleTemplate_):
    config = {
        'all_channels': {
            'default': True,
            'range': [True, False]
        },
        'init_value': {
            'default': 0.25,
        }
    }

    def __init__(self, in_shape=None, **kwargs):
        self.torch_name = 'PReLU'
        super().__init__(in_shape = in_shape, **kwargs)

    def _create_torch_args(self):
        if self.params['all_channels']:
            if type(self.in_shape) is int:
                num_channels = self.in_shape
            else:
                num_channels = self.in_shape[0]
        else:
            num_channels = 1
        self._args = []
        self._kwargs = {
            'num_parameters' : num_channels,
            'init' : self.params['init_value']
        }


class DropoutTmpl(ModuleTemplate_):
    config = {
        'p': {
            'default': 0.75,
            'range': list(np.linspace(0.05, 0.95, num=19))
        },
        'inplace': {
            'default': False,
        }
    }

    def __init__(self, in_shape=None, **kwargs):
        super().__init__(in_shape=in_shape, **kwargs)
        self.torch_name = 'Dropout'

    def _create_torch_args(self):
        self._args = [self.params['p']]
        self._kwargs = {}

    def __repr__(self):
        p = round(self.params['p'], 2)
        repr_str = '{:12} {:14} {:8}'.format( self.torch_name,
                                        str(self.out_shape),
                                        '(' + str(p) + ')' )
        return repr_str


class Dropout2dTmpl(DropoutTmpl):
    def __init__(self, in_shape=None, **kwargs):
        super().__init__(in_shape=in_shape, **kwargs)
        self.torch_name = 'Dropout2d'


class FlattenTmpl(ModuleTemplate_):
    def __init__(self, in_shape=None, **kwargs):
        self.torch_name = 'Flatten'
        super().__init__(in_shape=in_shape, **kwargs)

    def _update_out_shape(self):
        if self.in_shape is None:
            self.out_shape = None
        else:
            self.out_shape = 1
            # print(self.in_shape, type(self.in_shape))
            for dim in self.in_shape:
                self.out_shape *= dim

    # def instantiate_module(self):
    #     self._create_torch_args()
    #     klass = globals()[self.torch_name]
    #     instance = klass(*self._args, **self._kwargs)
    #     return instance


class ConvTemplate_(ModuleTemplate_):
    """
    Parent template for Convolutional layers.

    Children should implement _create_args() method, create config_data dictionary and
    call _conv_update_out_shape() with arguments specific for each layer type.

    Input:
     - in_shape (C, H, W): 3d tuple of number of input channels (like 3 RGB images),
        height and width
     - kwargs: key argumets specific for each layer
    """
    config = {
        'out_channels': {
            'default': 32,
            'range': [2, 4, 8],
            'learnable': True,
        },
        'kernel_size': {
            'default': (3, 3),
            'range': [(1, 1), (2, 2), (3, 3)],
            'learnable': True,
        },
        'stride': {
            'default': (1, 1),
            'range': [(1, 1), (2, 2), (3, 3)],
            'learnable': True,
        },
        'padding': {
            'default': True,
            'range': [True, False],
            'learnable': True,
        },
        'dilation': {
            'default': (1, 1),
            'learnable': False,
        },
        'ceil_mode': {
            'default': False,
            'learnable': False,
        }
    }

    # children should override this
    def _update_out_shape(self):
        self._conv_update_out_shape()

    def _conv_update_out_shape(self,
                               out_channels=None,
                               kernel_size=None,
                               stride=None,
                               dilation=None,
                               ceil_mode=None):
        """
        General method to calculate convolution output shape.

        If None args specified - defalult values taken from self.params

        Output:
        - None: updates self.out_shape
        """

        if self.in_shape is None:
            self.out_shape = None
            return
        else:
            in_shape = self.in_shape
            padding = self._calc_padding_size()
            if out_channels is None:
                out_channels = self.params['out_channels']
            if kernel_size is None:
                kernel_size = self.params['kernel_size']
            if stride is None:
                stride = self.params['stride']
            if dilation is None:
                dilation = self.params['dilation']
            if ceil_mode is None:
                ceil_mode = self.params['ceil_mode']

            out_shape_height = (in_shape[1] + 2*padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
            out_shape_width = (in_shape[2] + 2*padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1

            if ceil_mode:
                out_shape_height = ceil(out_shape_height)
                out_shape_width = ceil(out_shape_width)
            else:
                out_shape_height = floor(out_shape_height)
                out_shape_width = floor(out_shape_width)

            self.out_shape = (out_channels, out_shape_height, out_shape_width)

    def _calc_padding_size(self):
        """
        Calculate padding size for convolutional layers.

        Output:
        - (2d tuple of int): padding size
        """
        padding = self.params['padding']
        kernel_size = self.params['kernel_size']
        if padding:
            return (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            return (0, 0)


class Conv2dTmpl(ConvTemplate_):
    config = {
        'out_channels': {
            'default': 32,
            'range': [8, 16, 32, 64, 96, 128, 192]
        },
        'kernel_size': {
            'default': (3, 3),
            'range': [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]
        },
        'stride': {
            'default': (1, 1),
            'range': [(1, 1), (2, 2), (3, 3), (4, 4)]
        },
        'padding': {
            'default': True,
            'range': [True, False]
        },
        'padding_mode': {
            'default': 'zeros'
        },
        'dilation': {
            'default': (1, 1),
        },
        'groups': {
            'default': 1,
        },
        'bias': {
            'default': True,
        }
    }

    def __init__(self, in_shape=None, **kwargs):
        """

        """

        self.torch_name = 'Conv2d'
        super().__init__(in_shape=in_shape, **kwargs)
        # utils.print_nonprivate_properties(self)

    def _update_out_shape(self):
        self._conv_update_out_shape(ceil_mode=False)

    # def _conv_update_out_shape(self):
        # if self.in_shape is None:
        #     self.out_shape = None
        # else:
        #     in_shape = self.in_shape
        #     padding = self.params['padding_size']
        #     dilation = self.params['dilation']
        #     kernel_size = self.params['kernel_size']
        #     stride = self.params['stride']

        #     out_shape_height = (in_shape[1] + 2*padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        #     out_shape_width = (in_shape[2] + 2*padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

        #     self.out_shape = (self.params['out_channels'], out_shape_height, out_shape_width)

    def calc_maccs(self):
        self._update_out_shape()
        if not self.out_shape:
            kernel = self.params['kernel_size']
            self.maccs = kernel[0] * kernel[1] * self.in_shape[0] * \
                self.out_shape[0] * self.out_shape[1] * self.out_shape[2]
        else:
            self.maccs = 0

    def _calc_flops(self):
        kernel = self.params['kernel_size']
        bias = self.params['bias']
        if bias:
            self.flops = 2 * kernel[0] * kernel[1] * self.in_shape[0] * \
                self.out_shape[0] * self.out_shape[1] * self.out_shape[2]
        else:
            self.flops = (2 * kernel[0] * kernel[1] - 1) * self.in_shape[0] * \
                self.out_shape[0] * self.out_shape[1] * self.out_shape[2]

    def _create_torch_args(self):
        self._args = [
            self.in_shape[0],
            self.out_shape[0],
            self.params['kernel_size']
        ]
        self._kwargs = {
            'stride': self.params['stride'],
            'padding': self._calc_padding_size(),
            'dilation': self.params['dilation'],
            'groups': self.params['groups'],
            'bias': self.params['bias'],
            'padding_mode': self.params['padding_mode'],
        }

    def __repr__(self):
        kernel_size = str(self.params['kernel_size'])
        stride = str(self.params['stride'])
        repr_str = '{:12} {:14} {:8} {:8}'.format(self.torch_name,
                                                  str(self.out_shape),
                                                  kernel_size,
                                                  stride)
        return repr_str


class MaxPool2dTmpl(ConvTemplate_):

    config = {
        'kernel_size': {
            'default': (2, 2),
            'range': [(2, 2), (3, 3), (4, 4)]
        },
        'stride': {
            'default': (2, 2),
            'range': [(1, 1), (2, 2), (3, 3)]
        },
        'padding': {
            'default': False,
            'range': [True, False]
        },
        'dilation': {
            'default': (1, 1),
        },
        'return_indices': {
            'default': False,
        },
        # ceil mode doesn't work correctly TODO: report bug
        'ceil_mode': {
            'default': False,
        }
    }

    def __init__(self, in_shape=None, **kwargs):
        self.torch_name = 'MaxPool2d'
        super().__init__(in_shape = in_shape, **kwargs)
        # utils.print_nonprivate_properties(self)

    def _update_out_shape(self):
        self._conv_update_out_shape(out_channels=self.in_shape[0])

    # def _conv_update_out_shape(self):
        # if self.in_shape is None:
        #     self.out_shape = None
        # else:
        #     in_shape = self.in_shape
        #     padding = self.params['padding_size']
        #     dilation = self.params['dilation']
        #     kernel_size = self.params['kernel_size']
        #     stride = self.params['stride']
        #     ceil_mode = self.params['ceil_mode']

        #     out_shape_height = (in_shape[1] + 2*padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
        #     out_shape_width = (in_shape[2] + 2*padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
        #     if ceil_mode:
        #         out_shape_height = ceil(out_shape_height)
        #         out_shape_width = ceil(out_shape_width)
        #     else:
        #         out_shape_height = floor(out_shape_height)
        #         out_shape_width = floor(out_shape_width)

        #     self.out_shape = (in_shape[0], out_shape_height, out_shape_width)

    def _create_torch_args(self):
        self._args = [self.params['kernel_size']]
        self._kwargs = {'stride' : self.params['stride'],
                       'padding' : self._calc_padding_size(),
                       'dilation' : self.params['dilation'],
                       'ceil_mode' : self.params['ceil_mode'],
                      }

    def __repr__(self):
        kernel_size = str(self.params['kernel_size'])
        stride = str(self.params['stride'])
        repr_str = '{:12} {:14} {:8} {:8}'.format(self.torch_name,
                                                  str(self.out_shape),
                                                  kernel_size,
                                                  stride)
        return repr_str


class AvgPool2dTmpl(ConvTemplate_):
    config = {
        'kernel_size': {
            'default': (2, 2),
            'range': [(2, 2), (3, 3), (4, 4)]
        },
        'stride': {
            'default': (2, 2),
            'range': [(1, 1), (2, 2)]
        },
        'padding': {
            'default': False,
            'range': [True, False]
        },
        # ceil_mode=True doesn't work correctly TODO: report bug
        'ceil_mode': {
            'default': False,
        },
        'count_include_pad': {
            'default': True,
            'range': [True, False]
        },
    }

    def __init__(self, in_shape=None, **kwargs):
        self.torch_name = 'AvgPool2d'
        super().__init__(in_shape = in_shape, **kwargs)
        # utils.print_nonprivate_properties(self)

    def _update_out_shape(self):
        self._conv_update_out_shape(out_channels=self.in_shape[0],
                                    dilation=(1, 1))

    # def _conv_update_out_shape(self):
        # if self.in_shape is None:
        #     self.out_shape = None
        # else:
        #     in_shape = self.in_shape
        #     padding = self.params['padding_size']
        #     kernel_size = self.params['kernel_size']
        #     stride = self.params['stride']
        #     ceil_mode = self.params['ceil_mode']

        #     out_shape_height = (in_shape[1] + 2*padding[0] - kernel_size[0]) / stride[0] + 1
        #     out_shape_width = (in_shape[2] + 2*padding[1] - kernel_size[1]) / stride[1] + 1
        #     if ceil_mode:
        #         out_shape_height = ceil(out_shape_height)
        #         out_shape_width = ceil(out_shape_width)
        #     else:
        #         out_shape_height = floor(out_shape_height)
        #         out_shape_width = floor(out_shape_width)

        #     self.out_shape = (in_shape[0], out_shape_height, out_shape_width)

    def _create_torch_args(self):
        self._args = [self.params['kernel_size']]
        self._kwargs = {'stride' : self.params['stride'],
                       'padding' : self._calc_padding_size(),
                       'ceil_mode' : self.params['ceil_mode'],
                       'count_include_pad' : self.params['count_include_pad']
                      }

    def __repr__(self):
        kernel_size = str(self.params['kernel_size'])
        stride = str(self.params['stride'])
        repr_str = '{:12} {:14} {:8} {:8}'.format(self.torch_name,
                                                  str(self.out_shape),
                                                  kernel_size,
                                                  stride)
        return repr_str


class GlobalAvgPool2dTmpl(ModuleTemplate_):
    config = {}

    def __init__(self, in_shape=None):
        self.torch_name = 'AvgPool2d'
        super().__init__(in_shape=in_shape)

    def _update_out_shape(self):
        if self.in_shape is None:
            self.out_shape = None
        else:
            self.out_shape = (self.in_shape[0], 1, 1)    # out_shape is (num_channels, 1, 1)

    def _create_torch_args(self):
        # kernel_size = in_shape for AvgPool2d
        self._args = [(self.in_shape[1], self.in_shape[2])]
        self._kwargs = {}

    def __repr__(self):
        repr_str = '{:12} {:14}'.format('Global' + self.torch_name, str(self.out_shape))
        return repr_str





# class SequentialEx(Sequential):
#     def __init__(self, in_shape, out_shape, *templates):
#         self.in_shape = in_shape
#         self.module_templates = list(templates)
#         self._sync_template_shapes(in_shape, out_shape)
#         self._instantiate_modules()
#         super().__init__(*self.modules_instances)
#
#     def __repr__(self):
#         # We treat the extra repr like the sub-module, one item per line
#         extra_lines = []
#         extra_repr = self.extra_repr()
#         # empty string will be split into list ['']
#         if extra_repr:
#             extra_lines = extra_repr.split('\n')
#         child_lines = []
#         mod_count = 0
#         for key, module in self._modules.items():
#
#             mod_str = '{:12}'.format(module.__class__.__name__)
#             # mod_str += repr(module)
#
#             mod_shape = str(self.module_templates[mod_count].out_shape)
#             mod_shape = '{:12}'.format(mod_shape)
#             mod_str += mod_shape
#
#             # mod_params = sum(p.numel() for p in module.params() if p.requires_grad)
#             # params_str = '' + '{:3.1e}'.format(mod_params) if mod_params > 0 else ''
#             # params_str = _addindent(params_str, 7)
#             # mod_str += params_str
#
#             mod_count += 1
#             child_lines.append('{:>4}: '.format('('+key+')') + mod_str)
#         lines = extra_lines + child_lines
#
#         main_str = self._get_name() + '('
#         if lines:
#             # simple one-liner info, which most builtin Modules will use
#             if len(extra_lines) == 1 and not child_lines:
#                 main_str += extra_lines[0]
#             else:
#                 main_str += '\n  ' + '\n  '.join(lines) + '\n'
#
#         main_str += ')'
#         return main_str
#
#     def _sync_template_shapes(self, in_shape, out_shape):
#         for i, template in enumerate(self.module_templates):
#             # first elem
#             if i == 0:
#                 template.in_shape = in_shape
#             # last elem
#             elif i == len(self.module_templates) - 1:
#                 template.in_shape = previous_out_shape
#                 template.set_out_shape(out_shape)
#             # middle elem
#             else:
#                 template.in_shape = previous_out_shape
#             previous_out_shape = template.out_shape
#             self.module_templates[i] = template
#
#     def _instantiate_modules(self):
#         modules_instances = []
#         for template in self.module_templates:
#             # debug info
#             # print(template.torch_name, 'in_shape:', template.in_shape, 'out_shape:', template.out_shape)
#             module = template.instantiate_module()
#             module.in_shape = template.in_shape
#             modules_instances.append(module)
#         self.modules_instances = modules_instances


pass
################ LAYER TEMPLATES ####################


class LayerTemplate_():
    """
        Create Layer - main module template followed by auxilaries:
        Ex. Conv - DropOut - MaxPool - ReLU
        Two ways of creation:
        - From preset of templates passed as args:

            conv_layer = ConvLayerTmpl((3, 32, 32), tmpl1, tmpl2, tmpl3)

        - Randomly generated  main template followed by auxilaries:

            conv_layer = ConvLayerTmpl((3, 32, 32))
            conv_layer.gen_rand_layer()


        Each layer should have an activation function (in order to break linearity) randomly selected from
        config['activation'] list. Additionally, layer could have aux modules, with indicated
        frequency of appearance in config['freq']

        Upon creation only needs 'in_shape'. 'out_shape' is calculated (upon generation) or set from
        last template provided. If generated from templates - their shapes are synced:
        layer.in_shape -> tmpl1.in_shape -> tmpl2.in_shape = tmpl1.out_shape -> ... -> layer.out_shape = last_tmpl.out_shape

        If one of aux or main templates rendered zero in out_shape (ex. (32, 0, 0)), the flag
        self.is_zero_shape is set and further template generation stops.

        Example config for Convolution Layer Template

        config = {
            'activation': ['ReLUTmpl', 'LeakyReLUTmpl', 'PReLUTmpl'],
            'freq': {
                'MaxPool2dTmpl': 0.5,
                'BatchNorm2dTmpl': 0.5,
                'AvgPool2dTmpl': 0.5,
                'Dropout2dTmpl': 0
            }
        }


    """

    config = {}

    def __init__(self, in_shape=None, *templates):
        self.templates = list(templates)
        self._in_shape = in_shape
        self.out_shape = None
        self.is_zero_shape = False
        self.sync_shapes()

    @property
    def in_shape(self):
        return self._in_shape

    @in_shape.setter
    def in_shape(self, in_shape):
        self._in_shape = in_shape
        self.sync_shapes()

    def sync_shapes(self):
        """
        Syncs the shapes of inner templates when layer

        is not empty (contains templates) and in_shape is not None
        """

        if self.in_shape is None:
            self.out_shape = None
            return
        if not self.templates:
            return
        self._sync_shapes()

    def _sync_shapes(self):
        next_in_shape = self.in_shape
        for tmpl in self.templates:
            tmpl.in_shape = next_in_shape
            if tmpl.is_zero_shape:
                # debug info
                # print(tmpl.torch_name, tmpl.in_shape, tmpl.out_shape)
                self.templates.remove(tmpl)
            else:
                next_in_shape = tmpl.out_shape

        self.out_shape = next_in_shape

    def gen_rand_layer(self):
        self.templates = []
        self._generate_main_template()
        self._generate_aux_templates()
        self._shuffle_aux_templates()

    def _generate_main_template(self):
        '''
        Children should overwrite this method
        '''
        raise NotImplementedError

    def _generate_aux_templates(self):
        if self.config.get('activation') is None:
            return

        class_name = random.choice(self.config['activation'])
        self._generate_template(class_name, in_shape=self.out_shape)

        if self.config.get('freq') is None:
            return

        for name, freq in self.config['freq'].items():
            if random.random() < freq:
                self._generate_template(name, in_shape=self.out_shape)
                if self.is_zero_shape:
                    break

    def _shuffle_aux_templates(self):
        random.shuffle(self.templates[1:])

    def _generate_template(self, tmpl_class_name, in_shape):
        # appends template and updates self.out_shape
        tmpl = _instance_from_name(tmpl_class_name, in_shape=in_shape)
        tmpl.gen_rand_params()
        # print(tmpl.torch_name, tmpl.in_shape, tmpl.out_shape)
        if tmpl.is_zero_shape:
            # print(tmpl.torch_name + ': ZERO SHAPE', tmpl.out_shape)
            self.is_zero_shape = True
            self.out_shape = in_shape
        else:
            self.templates.append(tmpl)
            self.out_shape = tmpl.out_shape

    def instantiate_layer(self):
        modules = []
        for tmpl in self.templates:
            m = tmpl.instantiate_module()
            modules.append(m)
        return Sequential(*modules)

    def __repr__(self):
        out_str = ''
        for tmpl in self.templates:
            out_str += repr(tmpl) + '\n'            
        return out_str        


class ConvLayerTmpl(LayerTemplate_):
    '''
    Starts with main Conv2dTmpl template followed by combination of 
    auxilary templates from self.freq_dict
    '''

    config = {
        'activation': ['ReLUTmpl', 'LeakyReLUTmpl', 'PReLUTmpl'],
        'freq': {
            'MaxPool2dTmpl': 0.5,
            'BatchNorm2dTmpl': 0.5,
            'AvgPool2dTmpl': 0.5,
            'Dropout2dTmpl': 0,
        }
    }

    def _generate_main_template(self):
        self._generate_template('Conv2dTmpl', in_shape=self.in_shape)


class LinLayerTmpl(LayerTemplate_):
    config = {
        'activation': ['ReLUTmpl', 'LeakyReLUTmpl', 'PReLUTmpl'],
        'freq': {'DropoutTmpl': 0.5},
    }

    def __init__(self, in_shape=None, *templates):
        super().__init__(in_shape, *templates)

    def _generate_main_template(self):
        self._generate_template('LinearTmpl', in_shape=self.in_shape)


class LastLinLayerTmpl(LayerTemplate_):
    config = {}

    def __init__(self, in_shape=None, *templates):
        super().__init__(in_shape, *templates)

    def _generate_main_template(self):
        self._generate_template('LinearTmpl', in_shape = self.in_shape)


class GapLayerTmpl(LayerTemplate_):
    '''
        Global Average Pool layer. Single template
    '''
    def __init__(self, in_shape=None):
        super().__init__(in_shape)
    
    def gen_rand_layer(self):
        self.templates = []
        self._generate_main_template()

    def _generate_main_template(self):
        self._generate_template('GlobalAvgPool2dTmpl', in_shape=self.in_shape)


class FlatLayerTmpl(LayerTemplate_):
    def __init__(self, in_shape=None):
        super().__init__(in_shape)
    
    def gen_rand_layer(self):
        self._generate_main_template()

    def _generate_main_template(self):
        self._generate_template('FlattenTmpl', in_shape=self.in_shape)


####################### MODEL TEMPLATE ##################


class ModelLayersTmpl:
    '''
        Model created from Layer templates
    '''
    def __init__(self, in_shape=None, out_shape=None, *layers: LayerTemplate_):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.layers = list(layers)
        self.sync_shapes()

    def sync_shapes(self):
        if self.in_shape is None:
            self.out_shape = None
            return
        if not self.layers:
            return
        self._sync_shapes()

    def _sync_shapes(self):
        next_in_shape = self.in_shape
        for layer in self.layers:
            layer.in_shape = next_in_shape
            if layer.is_zero_shape:
                # print(layer, 'ZERO SHAPE')
                self.layers.remove(layer)
            else:
                next_in_shape = layer.out_shape
        self._set_last_shape()
        
    def _set_last_shape(self):
        self.layers[-1].templates[-1].set_out_shape(self.out_shape)

    def instantiate_model(self):
        mods = []
        for layer in self.layers:
            for tmpl in layer.templates:
                mods.append(tmpl.instantiate_module())
        model = Sequential(*mods)
        return model

    def gen_hash(self):
        hash_str = ''
        for layer in self.layers:
            for tmpl in layer.templates:
                hash_str += tmpl.gen_hash()        
        self.hash_str = hash_str

    def get_templates(self):
        templates = []
        for layer in self.layers:
            for tmpl in layer.templates:
                templates.append(tmpl)
        return templates

    def get_conv_part(self):
        conv_part = []
        for layer in self.layers:
            layer_name = layer.__class__.__name__
            if layer_name == 'ConvLayerTmpl':
                conv_part.append(layer)
        return conv_part

    def set_conv_part(self, conv_part):
        self._del_conv_part()
        self._add_conv_part(conv_part)
        self.sync_shapes()

    def _del_conv_part(self):
        for layer in reversed(self.layers):
            layer_name = layer.__class__.__name__
            if layer_name == 'ConvLayerTmpl':
                self.layers.remove(layer)
    
    def _add_conv_part(self, conv_part):
        self.layers = conv_part + self.layers

    def get_lin_part(self):
        lin_part = []
        for layer in self.layers:
            layer_name = layer.__class__.__name__
            if layer_name == 'LinLayerTmpl':
                lin_part.append(layer)
        return lin_part

    def __repr__(self):
        out_str = ''
        for layer in self.layers:
            out_str += repr(layer)          
        return out_str


class ModelTmpl:
    '''
        ModelTmpl created from Module templates. Last template must be linear.        
    '''
    def __init__(self, in_shape=None, out_shape=None, *templates: ModuleTemplate_):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.is_zero_shape = False
        self.templates = list(templates)
        self.sync_shapes()

    def sync_shapes(self):
        """
        Syncs the shapes of inner templates when model is not empty:

         - contains templates
         - in_shape and out_shape is not None
        """
        
        if self.in_shape is None:
            self.out_shape = None
            return
        if not self.templates:
            return
        self._sync_shapes()

    def _sync_shapes(self):
        in_shape = self.in_shape
        for tmpl in self.templates:
            tmpl.in_shape = in_shape
            tmpl.update_shape()
            if tmpl.is_zero_shape:
                # debug info
                print(tmpl.torch_name, tmpl.in_shape, tmpl.out_shape)
                raise ValueError(f'Zero in output shape template: {tmpl} shape: {tmpl.out_shape}')
                # self.templates.remove(tmpl)
            else:
                in_shape = tmpl.out_shape
        self._set_last_shape()

    def _set_last_shape(self):
        self.templates[-1].set_out_shape(self.out_shape)
        
    # def set_shapes(self, in_shape, out_shape):
        # self.in_shape = in_shape
        # self.out_shape = out_shape
        # self.sync_shapes()
    
    def instantiate_model(self, gpu=False):
        mods = []
        for tmpl in self.templates:
            mods.append(tmpl.instantiate_module())        
        model = Sequential(*mods)

        if gpu:
            return model.type(torch.cuda.FloatTensor)

        return model

    def get_conv_len(self):
        conv_layers = [t for t in self.templates if isinstance(t, Conv2dTmpl)]
        return len(conv_layers)

    def __len__(self):
        return len(self.templates)

    def __repr__(self):
        out_str = ''
        for tmpl in self.templates:
            out_str += repr(tmpl) + '\n'         
        return out_str

    def __eq__(self, other):
        if len(self.templates) != len(other.templates):
            return False
        for i in range(len(self.templates)):
            if self.templates[i] != other.templates[i]:
                return False
        return True


