import torch.nn as nn
import random
import numpy as np
from math import ceil, floor
import sophius.utils as utils
from collections import defaultdict


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # pylint: disable=unused-variable
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class Parameter:
    """         
        - value (same type): value of parameter
        - learnable (bool): is going to change during genetic search
        - on_change (function): triggers when parameter.value changes 
    """
    def __init__(self, value, learnable=True):
        self.value = value
        self.learnable = learnable

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key != 'on_change' and key == 'value':
            self.on_change()
                
    def on_change(self):
        # override this
        pass
        # raise NotImplementedError

    def __repr__(self):
        repr_str = str(self.value)
        return repr_str


class ConfigDict(dict):
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key != 'on_change':
            self.on_change()
                
    def on_change(self):
        # override this
        raise NotImplementedError


class ConfigGenerator:
    """
    Generates config for a given template instance.
    """
    def __init__(self, template):
        """
        Input:
        - template(ModuleTemplate_): instance of corresponding Module template
        """
        self.template = template

    def get(self, **kwargs):
        """
        Generate config dictionary from input arguments or if absent from defaults
        
        Input:
        - kwargs (dict): arguments with module specific config,
                ex: kernel_size = (3, 3), stride = (1, 1)
        Output:
        - get_config (dict): Generated config, 'Parameter' type dictionary
        """        
        config_data = self.template.config_data
        get_config = {}
        for key in config_data:
            value = kwargs.pop(key, config_data[key]['default'].value)
            learn = config_data[key]['default'].learnable
            get_config[key] = Parameter(value, learn)
        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)
        return get_config

    def get_random(self):
        """
        Randomly generate config dictionary from default ranges

        Output:
        - get_config (dict): Generated config, 'Parameter' type dictionary
        """        
        config_data = self.template.config_data
        get_config = {}
        for key in config_data:
            try:
                config_range = config_data[key]['range']
                get_config[key] = Parameter(random.choice(config_range), learnable=True)
            except KeyError:
                get_config[key] = config_data[key]['default']
        return get_config


###################### MODULE TEMPLATES ########################
class ModuleTemplate_():
    '''
        Base Class for Module Template.
        Serves as a template for future Pytorch Module instance. Holds only the necessary config of the 
        future module and doesn't hold weights that's why much lighter than Pytorch module. Could be then used a part of 
        Model template - architechture of future Model.
        
        Upon __init__(self, in_shape=None, **kwargs) 'in_shape' is always present, **kwargs
        are specific to each module and saved as self.config dict (Parameter class). Also for each config we have to set up
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
            model = nn.Sequential(conv, flatten, lin)

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
                models.append(nn.Sequential(*mods))
    '''

    config_data = {}
    '''        
    Contains as set of keyword parameters for module. The Parameter type defines if the parameters 
    is learnable or not. If learnable (True by default) then a range of possible values should be
    provided.

    Example for LinearModule Template:

    config_data = {
        'out_features': {
            'default': Parameter(256),
            'range': [32, 64, 128, 256, 512, 1024, 2048, 4096]
        },
        'bias': {
            'default': Parameter(True, learnable=False)
        }
    }
    '''

    # pytorch module name
    module_name = None

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
        self._config = ConfigGenerator(self).get(**kwargs)
        # config change handlers
        # self._config.on_change = self.update_shape
        # for key in self._config:
        #     self._config[key].on_change = self.update_shape
        self.update_shape()
        # utils.print_nonprivate_properties(self)

    @property
    def config(self):
        return self._config
    
    @config.setter
    def config(self, value):
        self._config = value
        self.update_shape()

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
        self.out_shape = self.in_shape

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
        # scalar shape
        elif self.out_shape < 1:
            self.is_zero_shape = True

    def set_out_shape(self, out_shape):
        """
        We can set out_shape only for Linear type templates.        
        """
        # Linear templates overwrites this
        raise Exception('Cannot set out_shape for Non linear templates (%s)' % self.module_name)

    def calc_maccs(self):
        '''
        Multiply-ACCumulate operationS
        Children override this
        '''
        self.maccs = 0

    def calc_flops(self):
        if self.out_shape:
            self._calc_flops()
        else:
            self.flops = 0

    def _calc_flops(self):
        '''
        Flops calculation for each module type. Children will override this
        '''
        self.flops = 0

    def gen_rand_config(self):
        self.config = ConfigGenerator(self).get_random()

    def get_learnable_params(self):
        """
        Returns a list of learnable params
        """
        res = {}
        for k, v in self.config_data.items():
            if v.get('default').learnable:
                res[k] = self.config[k]
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
        nn_instance = globals()['nn']
        klass = getattr(nn_instance, self.module_name)
        instance = klass(*self._args, **self._kwargs)
        return instance
        # else:
            # raise TypeError("in_shape should be int or tuple, instead {}".format(in_type))

    def _create_torch_args(self):
        '''
        Children overwrite this.

        Creates arguments from self.config for pytorch Module class
        Because sometimes there is no direct correspond between Template args
        and Pytorch Module args.
        
        Example for pytorch Linear(in_shape, features, bias = True):
            self._args = [self.in_shape, self.out_shape]
            self._kwargs = {'bias' : self.config['bias'].value}
                    
        '''
        self._args = []
        self._kwargs = {}

    def __repr__(self):
        # tmpl_str = ', '.join((repr(self.in_shape), repr(self.config)))
        repr_str = '{:12} {:14}'.format(str(self.module_name), str(self.out_shape))
        return repr_str

    def __eq__(self, other):
        equal = True       
        # check different modules
        if self.module_name != other.module_name:
            equal = False
            return equal
        # check config for same modules
        for key in self.config:
            self_val = self.config[key].value
            try:
                other_val = other.config[key].value
                if self_val != other_val:
                    equal = False
                    break
            except KeyError:
                equal = False
                break
        return equal


class LinearTmpl(ModuleTemplate_):

    config_data = {
        'out_features': {
            'default': Parameter(256),
            'range': [32, 64, 128, 256, 512, 1024, 2048, 4096]
        },
        'bias': {
            'default': Parameter(True, learnable=False)
        }
    }
    
    def __init__(self, in_shape=None, **kwargs):
        self.module_name = 'Linear'
        super().__init__(in_shape=in_shape, **kwargs)
        # utils.print_nonprivate_properties(self)
    
    def _update_out_shape(self):
        self.out_shape = self.config['out_features'].value

    def set_out_shape(self, out_shape):
        self.out_shape = out_shape
        self.config['out_features'] = Parameter(out_shape)

    def calc_maccs(self):
        self.maccs = self.in_shape * self.out_shape

    def _calc_flops(self):
        bias = self.config['bias'].value
        if bias:
            self.flops = 2 * self.in_shape * self.out_shape
        else:
            self.flops = (2 * self.in_shape - 1) * self.out_shape

    def _create_torch_args(self):
        self._args = [self.in_shape, self.config['out_features'].value]
        self._kwargs = {'bias': self.config['bias'].value}


class BatchNorm2dTmpl(ModuleTemplate_):
    config_data = {
        'eps': {
            'default': Parameter(1e-5, learnable=False)
        },
        'momentum': {
            'default': Parameter(0.1, learnable=False)
        },
        'affine': {
            'default': Parameter(True, learnable=False)
        },
        'track_running_stats': {
            'default': Parameter(True, learnable=False)
        }
    }

    def __init__(self, in_shape=None, **kwargs):
        self.module_name = 'BatchNorm2d'
        super().__init__(in_shape = in_shape, **kwargs)
        # utils.print_nonprivate_properties(self)
    
    def _create_torch_args(self):
        self._args = [self.in_shape[0]]
        self._kwargs = {
            'eps' : self.config['eps'].value,
            'momentum' : self.config['momentum'].value,
            'affine' : self.config['affine'].value,
            'track_running_stats' : self.config['track_running_stats'].value
        }        

        
class ReLUTmpl(ModuleTemplate_):
    config_data = {
        'inplace': {
            'default': Parameter(False, learnable=False)
        }
    }

    def __init__(self, in_shape=None, **kwargs):
        self.module_name = 'ReLU'
        super().__init__(in_shape = in_shape, **kwargs)

    def _create_torch_args(self):
        self._args = []
        self._kwargs = {
            'inplace' : self.config['inplace'].value
        }


class LeakyReLUTmpl(ModuleTemplate_):
    config_data = {
        'negative_slope': {
            'default': Parameter(0.1),
            'range': [0.1, 0.01, 0.001]
        },
        'inplace': {
            'default': Parameter(False, learnable=False)
        }
    }

    def __init__(self, in_shape=None, **kwargs):
        self.module_name = 'LeakyReLU'
        super().__init__(in_shape = in_shape, **kwargs)
        
    def _create_torch_args(self):
        self._args = []
        self._kwargs = {
            'negative_slope': self.config['negative_slope'].value,
            'inplace': self.config['inplace'].value
        }

    def __repr__(self):
        negative_slope = self.config['negative_slope'].value
        repr_str = '{:12} {:14} {:8}'.format( self.module_name,
                                        str(self.out_shape),
                                        '(' + str(negative_slope) + ')' )
        return repr_str


class PReLUTmpl(ModuleTemplate_):
    config_data = {
        'all_channels': {
            'default': Parameter(True),
            'range': [True, False]
        },
        'init_value': {
            'default': Parameter(0.25, learnable=False)
        }
    }

    def __init__(self, in_shape=None, **kwargs):
        self.module_name = 'PReLU'
        super().__init__(in_shape = in_shape, **kwargs)
        
    def _create_torch_args(self):
        if self.config['all_channels'].value:
            if type(self.in_shape) is int:
                num_channels = self.in_shape
            else:
                num_channels = self.in_shape[0]
        else:
            num_channels = 1            
        self._args = []
        self._kwargs = {
            'num_parameters' : num_channels,
            'init' : self.config['init_value'].value
        }


class DropoutTmpl(ModuleTemplate_):
    config_data = {
        'p': {
            'default': Parameter(0.75),
            'range': list(np.linspace(0.05, 0.95, num=19))
        },
        'inplace': {
            'default': Parameter(False, learnable=False)
        }
    }

    def __init__(self, in_shape=None, **kwargs):
        super().__init__(in_shape=in_shape, **kwargs)
        self.module_name = 'Dropout'
    
    def _create_torch_args(self):
        self._args = [self.config['p'].value]
        self._kwargs = {}

    def __repr__(self):
        p = round(self.config['p'].value, 2)
        repr_str = '{:12} {:14} {:8}'.format( self.module_name,
                                        str(self.out_shape),
                                        '(' + str(p) + ')' )
        return repr_str


class Dropout2dTmpl(DropoutTmpl):
    def __init__(self, in_shape=None, **kwargs):
        super().__init__(in_shape=in_shape, **kwargs)
        self.module_name = 'Dropout2d'


class FlattenTmpl(ModuleTemplate_):
    def __init__(self, in_shape=None, **kwargs):
        self.module_name = 'Flatten'
        super().__init__(in_shape = in_shape, **kwargs)

    def _update_out_shape(self):
        if self.in_shape is None:
            self.out_shape = None
        else:
            self.out_shape = 1
            # print(self.in_shape, type(self.in_shape))
            for dim in self.in_shape:
                self.out_shape *= dim
    
    def instantiate_module(self):
        self._create_torch_args()
        klass = globals()[self.module_name]
        instance = klass(*self._args, **self._kwargs)
        return instance


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
    config_data = {
        'out_channels': {
            'default': Parameter(32),
            'range': [2, 4, 8]
        },
        'kernel_size': {
            'default': Parameter((3, 3)),            
            'range': [(1, 1), (2, 2), (3, 3)]
        },
        'stride': {
            'default': Parameter((1, 1)),
            'range': [(1, 1), (2, 2), (3, 3)]
        },
        'padding': {
            'default': Parameter(True),
            'range': [True, False]
        },
        'dilation': {
            'default': Parameter((1, 1))
        },
        'ceil_mode': {
            'default': Parameter(False)
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
        
        If None args specified - defalult values taken from self.config
        
        Output:
        - None: updates self.out_shape
        """
        
        if self.in_shape is None:
            self.out_shape = None
            return
        else:
            in_shape=self.in_shape
            padding = self._calc_padding_size()
            if out_channels is None:
                out_channels=self.config['out_channels'].value
            if kernel_size is None:
                kernel_size = self.config['kernel_size'].value
            if stride is None:
                stride = self.config['stride'].value
            if dilation is None:
                dilation = self.config['dilation'].value
            if ceil_mode is None:
                ceil_mode = self.config['ceil_mode'].value
            
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
        padding = self.config['padding'].value
        kernel_size = self.config['kernel_size'].value
        if padding:
            return (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            return (0, 0)


class Conv2dTmpl(ConvTemplate_): 

    config_data = {
        'out_channels': {
            'default': Parameter(32),
            'range': [8, 16, 32, 64, 96, 128, 192]
        },
        'kernel_size': {
            'default': Parameter((3, 3)),
            'range': [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]
        },
        'stride': {
            'default': Parameter((1, 1)),
            'range': [(1, 1), (2, 2), (3, 3), (4, 4)]
        },
        'padding': {
            'default': Parameter(True),
            'range': [True, False]
        },
        'padding_mode': {
            'default': Parameter('zeros', learnable=False)
        },
        'dilation': {
            'default': Parameter((1, 1), learnable=False)
        },
        'groups': {
            'default': Parameter(1, learnable=False)
        },
        'bias': {
            'default': Parameter(True, learnable=False)
        }
    }

    def __init__(self, in_shape=None, **kwargs):
        """
        
        """
        
        self.module_name = 'Conv2d'
        super().__init__(in_shape = in_shape, **kwargs)
        # utils.print_nonprivate_properties(self)

    def _update_out_shape(self):
        self._conv_update_out_shape(ceil_mode=False)

    # def _conv_update_out_shape(self):
        # if self.in_shape is None:
        #     self.out_shape = None
        # else:
        #     in_shape = self.in_shape
        #     padding = self.config['padding_size'].value
        #     dilation = self.config['dilation'].value
        #     kernel_size = self.config['kernel_size'].value
        #     stride = self.config['stride'].value

        #     out_shape_height = (in_shape[1] + 2*padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        #     out_shape_width = (in_shape[2] + 2*padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

        #     self.out_shape = (self.config['out_channels'].value, out_shape_height, out_shape_width)

    def calc_maccs(self):
        self._update_out_shape()
        if not self.out_shape:
            kernel = self.config['kernel_size'].value
            self.maccs = kernel[0] * kernel[1] * self.in_shape[0] * \
                self.out_shape[0] * self.out_shape[1] * self.out_shape[2]
        else:
            self.maccs = 0

    def _calc_flops(self):
        kernel = self.config['kernel_size'].value
        bias = self.config['bias'].value
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
            self.config['kernel_size'].value
        ]
        self._kwargs = {
            'stride': self.config['stride'].value,
            'padding': self._calc_padding_size(),
            'dilation': self.config['dilation'].value,
            'groups': self.config['groups'].value,
            'bias': self.config['bias'].value,
            'padding_mode': self.config['padding_mode'].value,
        }

    def __repr__(self):
        kernel_size = str(self.config['kernel_size'].value)
        stride = str(self.config['stride'].value)        
        repr_str = '{:12} {:14} {:8} {:8}'.format(self.module_name, 
                                                  str(self.out_shape), 
                                                  kernel_size,
                                                  stride)
        return repr_str        


class MaxPool2dTmpl(ConvTemplate_):

    config_data = {
        'kernel_size': {
            'default': Parameter((2, 2)),
            'range': [(2, 2), (3, 3), (4, 4)]
        },
        'stride': {
            'default': Parameter((2, 2)),
            'range': [(1, 1), (2, 2), (3, 3)]
        },
        'padding': {
            'default': Parameter(False),
            'range': [True, False]
        },
        'dilation': {
            'default': Parameter((1, 1), learnable = False)
        },
        'return_indices': {
            'default': Parameter(False, learnable = False)
        },
        # ceil mode doesn't work correctly TODO: report bug
        'ceil_mode': {
            'default': Parameter(False, learnable = False)
        }
    }

    def __init__(self, in_shape=None, **kwargs):
        self.module_name = 'MaxPool2d'
        super().__init__(in_shape = in_shape, **kwargs)
        # utils.print_nonprivate_properties(self)

    def _update_out_shape(self):
        self._conv_update_out_shape(out_channels=self.in_shape[0])

    # def _conv_update_out_shape(self):
        # if self.in_shape is None:
        #     self.out_shape = None
        # else:
        #     in_shape = self.in_shape
        #     padding = self.config['padding_size'].value
        #     dilation = self.config['dilation'].value
        #     kernel_size = self.config['kernel_size'].value
        #     stride = self.config['stride'].value
        #     ceil_mode = self.config['ceil_mode'].value

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
        self._args = [self.config['kernel_size'].value]
        self._kwargs = {'stride' : self.config['stride'].value,
                       'padding' : self._calc_padding_size(),
                       'dilation' : self.config['dilation'].value,                      
                       'ceil_mode' : self.config['ceil_mode'].value,
                      }

    def __repr__(self):
        kernel_size = str(self.config['kernel_size'].value)
        stride = str(self.config['stride'].value)        
        repr_str = '{:12} {:14} {:8} {:8}'.format(self.module_name, 
                                                  str(self.out_shape), 
                                                  kernel_size,
                                                  stride)
        return repr_str        


class AvgPool2dTmpl(ConvTemplate_):
    config_data = {
        'kernel_size': {
            'default': Parameter((2, 2)),
            'range': [(2, 2), (3, 3), (4, 4)]
        },
        'stride': {
            'default': Parameter((2, 2)),
            'range': [(1, 1), (2, 2)]
        },
        'padding': {
            'default': Parameter(False),
            'range': [True, False]
        },
        # ceil_mode=True doesn't work correctly TODO: report bug
        'ceil_mode': {
            'default': Parameter(False, learnable=False)
        },
        'count_include_pad': {
            'default': Parameter(True),
            'range': [True, False]
        },
    }

    def __init__(self, in_shape=None, **kwargs):
        self.module_name = 'AvgPool2d'
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
        #     padding = self.config['padding_size'].value
        #     kernel_size = self.config['kernel_size'].value
        #     stride = self.config['stride'].value
        #     ceil_mode = self.config['ceil_mode'].value

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
        self._args = [self.config['kernel_size'].value]
        self._kwargs = {'stride' : self.config['stride'].value,
                       'padding' : self._calc_padding_size(),
                       'ceil_mode' : self.config['ceil_mode'].value,
                       'count_include_pad' : self.config['count_include_pad'].value
                      }

    def __repr__(self):
        kernel_size = str(self.config['kernel_size'].value)
        stride = str(self.config['stride'].value)        
        repr_str = '{:12} {:14} {:8} {:8}'.format(self.module_name, 
                                                  str(self.out_shape), 
                                                  kernel_size,
                                                  stride)
        return repr_str        


class GlobalAvgPool2dTmpl(ModuleTemplate_):
    config_data = {}

    def __init__(self, in_shape=None):
        self.module_name = 'AvgPool2d'
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
        repr_str = '{:12} {:14}'.format('Global' + self.module_name, str(self.out_shape))
        return repr_str        


######################### EX - SEQUENTIAL ########################
    

class SequentialEx(nn.Sequential):    
    def __init__(self, in_shape, out_shape, *templates):
        self.in_shape = in_shape
        self.module_templates = list(templates)
        self._sync_template_shapes(in_shape, out_shape)
        self._instantiate_modules()
        super().__init__(*self.modules_instances)
        
    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        mod_count = 0
        for key, module in self._modules.items():
            
            mod_str = '{:12}'.format(module.__class__.__name__)
            # mod_str += repr(module)
            
            mod_shape = str(self.module_templates[mod_count].out_shape)
            mod_shape = '{:12}'.format(mod_shape)
            mod_str += mod_shape

            # mod_params = sum(p.numel() for p in module.config() if p.requires_grad)
            # params_str = '' + '{:3.1e}'.format(mod_params) if mod_params > 0 else ''
            # params_str = _addindent(params_str, 7)            
            # mod_str += params_str
            
            mod_count += 1
            child_lines.append('{:>4}: '.format('('+key+')') + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str
    
    def _sync_template_shapes(self, in_shape, out_shape):        
        for i, template in enumerate(self.module_templates):                        
            # first elem
            if i == 0:
                template.in_shape = in_shape
            # last elem
            elif i == len(self.module_templates) - 1:
                template.in_shape = previous_out_shape
                template.set_out_shape(out_shape)
            # middle elem
            else:
                template.in_shape = previous_out_shape
            previous_out_shape = template.out_shape
            self.module_templates[i] = template
            
    def _instantiate_modules(self):
        modules_instances = []
        for template in self.module_templates:             
            # debug info
            # print(template.module_name, 'in_shape:', template.in_shape, 'out_shape:', template.out_shape)
            module = template.instantiate_module()
            module.in_shape = template.in_shape
            modules_instances.append(module)
        self.modules_instances = modules_instances


################ LAYER TEMPLATES ####################


class LayerTemplate_():
    '''
        Create Layer - main module template followed by auxilaries:
        Ex. Conv - DropOut - MaxPool - ReLU
        Two ways of creation:
        - From preset of templates passed as args: 
         
            conv_layer = ConvLayerTmpl((3, 32, 32), tmpl1, tmpl2, tmpl3)
         
        - Randomly generated  main template followed by auxilaries:

            conv_layer = ConvLayerTmpl((3, 32, 32))
            conv_layer.gen_rand_layer()

        The list of aux'es and probabilities of their generation must be listed in self.freq_dict.
        All template config are generated randomly from internal lists of each
        template.
        
        Upon creation only needs 'in_shape'. 'out_shape' is calculated (upon generation) or set from 
        last template provided. If generated from templates - their shapes are synced:
        layer.in_shape -> tmpl1.in_shape -> tmpl2.in_shape = tmpl1.out_shape -> ... -> layer.out_shape = last_tmpl.out_shape

        If one of aux or main templates rendered zero in out_shape (ex. (32, 0, 0)), the flag
        self.is_zero_shape is set and further template generation stops.
    '''

    freq_dict = {}

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
        last_in_shape = self.in_shape
        for tmpl in self.templates:
            tmpl.in_shape = last_in_shape
            if tmpl.is_zero_shape:
                # print(tmpl.module_name, tmpl.in_shape, tmpl.out_shape)
                self.is_zero_shape = True
            last_in_shape = tmpl.out_shape
        self.out_shape = tmpl.out_shape

    # def set_in_shape(self, in_shape):
        # self.in_shape = in_shape
        # self.out_shape = in_shape
        # self.sync_shapes()

    def gen_rand_layer(self):
        self.templates = []
        self._generate_main_template()        
        self._generate_aux_templates()

    def _generate_main_template(self):
        '''
        Children should overwrite this method
        '''
        pass
        # raise NotImplementedError

    def _generate_aux_templates(self):
        freq_dict = utils.shuffle_dict(self.freq_dict)
        for tmpl_name in freq_dict:
            if tmpl_name == 'activation':
                tmpl_name = random.choice(freq_dict['activation'])
                self._generate_template(tmpl_name, in_shape=self.out_shape)
                if self.is_zero_shape:
                    break
            else:
                tmpl_freq = freq_dict[tmpl_name]
                if tmpl_freq > random.random():
                    self._generate_template(tmpl_name, in_shape=self.out_shape)
                    if self.is_zero_shape:
                        break

    def _generate_template(self, tmpl_name, in_shape):
        # appends template and updates self.out_shape
        tmpl = self._instantiate_template(tmpl_name, in_shape = in_shape)
        tmpl.gen_rand_config()
        # print(tmpl.module_name, tmpl.in_shape, tmpl.out_shape)
        if tmpl.is_zero_shape:
            # print(tmpl.module_name + ': ZERO SHAPE', tmpl.out_shape)
            self.is_zero_shape = True
            self.out_shape = in_shape
        else:
            self.templates.append(tmpl)
            self.out_shape = tmpl.out_shape
    
    def _instantiate_template(self, tmpl_name, **kwargs):
        klass = globals()[tmpl_name]
        template = klass(**kwargs)
        return template

    def instantiate_layer(self):
        module_instances = []
        for tmpl in self.templates:
            instance = tmpl.instantiate_module()
            module_instances.append(instance)
        return nn.Sequential(*module_instances)

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
    def __init__(self, in_shape=None, *templates):        
        self.freq_dict = {'activation': ['ReLUTmpl', 'LeakyReLUTmpl', 'PReLUTmpl'],
                        'MaxPool2dTmpl': 0.5,
                        'BatchNorm2dTmpl': 0.5,
                        'AvgPool2dTmpl': 0.5,
                        'Dropout2dTmpl': 0}
        super().__init__(in_shape, *templates)
            
    def _generate_main_template(self):
        self._generate_template('Conv2dTmpl', in_shape = self.in_shape)


class LinLayerTmpl(LayerTemplate_):
    def __init__(self, in_shape=None, *templates):
        self.freq_dict = {'activation': ['ReLUTmpl', 'LeakyReLUTmpl', 'PReLUTmpl'],
                          'DropoutTmpl': 0.5}
        super().__init__(in_shape, *templates)

    def _generate_main_template(self):
        self._generate_template('LinearTmpl', in_shape=self.in_shape)


class LastLinLayerTmpl(LayerTemplate_):
    def __init__(self, in_shape=None, *templates):
        self.freq_dict = {}
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
        self._generate_template('FlattenTmpl', in_shape = self.in_shape)


####################### MODEL TEMPLATE ##################


class ModelTmpl:
    '''
        Model created from Layer templates
    '''
    def __init__(self, in_shape=None, out_shape=None, *layers):
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
        model = nn.Sequential(*mods)
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


class ModelTmpl_:
    '''
        ModelTmpl created from Module templates. Last template must be linear.        
    '''
    def __init__(self, in_shape=None, out_shape=None, *templates):
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
        if self.in_shape is None:
            self.out_shape = None
            return        
        if not self.templates:
            return
        next_in_shape = self.in_shape
        for tmpl in self.templates:
            tmpl.in_shape = next_in_shape
            tmpl._update_out_shape()
            if tmpl._is_zero_shape():
                # debug info
                # print(tmpl.module_name, tmpl.in_shape, tmpl.out_shape)
                self.templates.remove(tmpl)
            else:
                next_in_shape = tmpl.out_shape
        self._set_last_shape()

    def _set_last_shape(self):
        self.templates[-1].set_out_shape(self.out_shape)
        
    # def set_shapes(self, in_shape, out_shape):
        # self.in_shape = in_shape
        # self.out_shape = out_shape
        # self.sync_shapes()
    
    def instantiate_model(self):
        mods = []
        for tmpl in self.templates:
            mods.append(tmpl.instantiate_module())        
        model = nn.Sequential(*mods)
        return model

    def __repr__(self):
        out_str = ''
        for tmpl in self.templates:
            out_str += repr(tmpl) + '\n'         
        return out_str

