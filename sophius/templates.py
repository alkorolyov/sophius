import torch.nn as nn
import random
import numpy as np
from math import ceil, floor
import sophius.utils as utils


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # pylint: disable=unused-variable
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class ParameterEx():
    '''
    Class for parameters of Templates
    
    'learnable' = True, defines whether parameter is going be optimized during genetic search
    also it is going to be initialized as random value from defined list of values
    
    'learnable' = False, default value will be used, not changed during genetic search

    '''
    def __init__(self, value, learnable = True):
        self.value = value
        self.learnable = learnable
    def __repr__(self):
        # repr_str = 'ParamEx[' + str(self.value) + ']'
        repr_str = str(self.value)
        return repr_str

'''
    Template (ModuleTmpl) -> Layer template (class LayerTemplate) -> Model template (class ModelTmpl)

'''

###################### MODULE TEMPLATES ########################


class ModuleTemplate_(): 
    '''
        Base Class for Module Template.
        Serves as a template for future Pytorch Module instance. Holds only the necessary parameters of the 
        future module and doesn't hold weights that's why much lighter than Pytorch module. Could be then used a part of 
        Model template - architechture of future Model.
        
        Upon __init__(self, in_shape=None, **params) 'in_shape' is always present, **params
        are specific to each module and saved as self.params dict (ParameterEx class). Also for each parameter we have to set up
        a range of acceptable values for further optimisation or set '.learnable = False' (self._init_param_values_list())
        When optimising parameters we can generate a new random set of parameters from corresponding range (self.gen_params)
        
        Important! There is no direct correspondance between ModuleTemplate_ Class parameters and PyTorch Module 
        Class parameters.

        After initialisation (__init__) self.in_shape attribute is determined and we can calculate self.out_shape 
        (self.calc_out_shape() method) for linkage with other templates when it will be used as a part of Model.

        Finally in order to get Pytorch module instance from template we need first create arguments for Pytorch Module, as not all
        Module Template args corresponds to Pytorch module args. (self._create_args(): self.params -> self.args, self.kwargs) 
        Template could be then instantiated into pytorch Module and used for training ( self.instantiate_module() ).

        Example N1:            
            # create simple 1 layer model from templates
            # with predetermined parameters
            # Conv - Flatten - Linear
            # create Conv2d template with CIFAR10 shape (3, 32, 32)        
            conv_tmpl = Conv2dTmpl(in_shape = (3, 32, 32), 
                                    kernel_size = 3, 
                                    stride = 1, 
                                    padding = True)
            
            # calculate the output shape
            conv_tmpl.calc_out_shape()
            
            # get pytorch module instance
            conv = conv_tmpl.instantiate_module()

            # repeat for Linear and Flatten tmpl
            flatten_tmpl = FlattenTmpl(in_shape = conv_tmpl.out_shape)
            flatten_tmpl.calc_out_shape()
            flatten = flatten_tmpl.instantiate_module()

            lin_tmpl = linTmpl(in_shape = flatten_tmpl.out_shape, out_shape = 10)
            lin_tmpl.calc_out_shape()
            lin = lin_tmpl.instantiate_module()
            
            # create model
            model = nn.Sequential(conv, flatten, lin)

        Example N2:

            # generate 10 models with random parameters using self.gen_params():
            
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
                    
                    # generate random params for each tmpl
                    tmpl.gen_params()
                    
                    tmpl.calc_out_shape()
                    next_in_shape = tmpl.out_shape
                
                # set out_shape 10 for the last tmpl
                templates[-1].set_out_shape(10)
                
                # list of pytoch modules in model
                mods =[]
                # create pytorch modules
                for tmpl in templates:
                    mods.append(tmpl.instantiate_module())

                # create pytorch model
                models.append(nn.Sequential(*mods))
    '''

    def __init__(self, in_shape=None, **params):
        self.module_name = 'Parent'
        self.in_shape = in_shape
        self.out_shape = None
        # utils.print_nonprivate_properties(self)
        self._init_params(params)
        self._init_param_values_list()

    def _init_params(self, params):
        '''
        Init params from **params dictionary to ParameterEx class
        '''
        
        self.params = {}        
        for key in params:
            self.params[key] = ParameterEx(params[key])        

    def _init_param_values_list(self):
        '''
        Children override this method
        Set's the range of possible values for each param

        ex: self.param_values['kernel_size'] = [(2, 2), (3, 3)]
            self.param_values['stride'] = [1, 2]
        '''
        self.param_values = {}
    
    def set_in_shape(self, in_shape):
        self.in_shape = in_shape
    
    # def set_out_shape(self, out_shape):
        # print(self.module_name + ': cannot set out_shape for this template')
            
    def calc_out_shape(self):
        self.out_shape = self.in_shape
    
    def check_zero_shape(self):
        is_zero_shape = False
        # multidim shape
        if type(self.out_shape) is tuple:
            for dim in self.out_shape:
                if dim < 1:
                    is_zero_shape = True

        # scalar shape
        elif self.out_shape < 1:
            is_zero_shape = True
        return is_zero_shape

    def calc_maccs(self):
        '''
            Multiply-ACCumulate operationS
            Children override this
        '''
        self.maccs = 0

    def calc_flops(self):
        self.calc_out_shape()
        if self.out_shape:
            self._calc_flops()
        else:
            self.flops = 0

    def _calc_flops(self):
        '''
            Flops calculation for each module type. Children will override this
        '''
        self.flops = 0

    def gen_params(self, random_gen = True):
        '''
        For each parameter generate a random value from a list self.param_values[]
        '''

        learnable_params = (key for key in self.params if self.params[key].learnable)
        if random_gen:            
            for key in learnable_params:
                # print(key, self.param_values[key])
                self.params[key].value = random.choice(self.param_values[key])
        else:
            # Not implemented
            pass

    def instantiate_module(self):
        in_type = type(self.in_shape)
        # if (in_type is int) or (in_type is tuple):
        self._create_args()
        torch_instance = globals()['nn']
        klass = getattr(torch_instance, self.module_name)
        instance = klass(*self.args, **self.kwargs)
        return instance
        # else:
            # raise TypeError("in_shape should be int or tuple, instead {}".format(in_type))

    # def gen_hash(self):
        # # module name to number:
        # hash_str = ''
        # module_ind = params_range_dict['module_name'].index(self.module_name)
        # hash_str += hash_dict['module_name'][module_ind]
        # # print(self.module_name, hash_str) # debug
        # # param value to number
        # for key in self.params:
        #     if self.params[key].learnable:                
        #         param_val = self.params[key].value
        #         param_ind = self.param_values[key].index(param_val)
        #         hash_str += hash_dict[key][param_ind]
        #         # print(key, hash_dict[key][param_ind])  # debug
        # return hash_str

    def _create_args(self):
        '''
        Children ovewrite this
        Creates arguments from self.params for pytorch Module class
        Example for pytorch Linear(in_shape, features, bias = True):
            self.args = [self.in_shape, self.out_shape]
            self.kwargs = {'bias' : self.params['bias'].value}
                    
        '''
        self.args = []
        self.kwargs = {}            
        
    # def _int2tuple_broadcast(self, param_name):
        # value = self.params[param_name].value
        # if type(value) is not tuple:
        #     self.params[param_name].value = (value, value)

    def __repr__(self):
        # tmpl_str = ', '.join((repr(self.in_shape), repr(self.params)))
        repr_str = '{:12} {:14}'.format(str(self.module_name), str(self.out_shape))
        return repr_str

    def __eq__(self, other):
        equal = True       
        # check different modules
        if self.module_name != other.module_name:
            equal = False
            return equal
        # check params for same modules
        for key in self.params:
            val1 = self.params[key].value
            try:            
                val2 = other.params[key].value
                if val1 != val2:
                    equal = False
                    break
            except:
                equal = False
                break
        return equal


class LinearTmpl(ModuleTemplate_):
    def __init__(self, in_shape=None, out_features=256, bias=True):        
        super().__init__(in_shape=in_shape,        
                         out_features=out_features, 
                         bias=bias)        
        self.module_name = 'Linear'
        self.calc_out_shape()
    
    def _init_param_values_list(self):
        self.param_values = {}
        self.param_values['out_features'] = [32, 64, 128, 256, 512, 1024, 2048, 4096]
        self.params['bias'].learnable = False
        
    def calc_out_shape(self):
        self.out_shape = self.params['out_features'].value

    def set_out_shape(self, out_shape):
        self.out_shape = out_shape
        self.params['out_features'] = ParameterEx(out_shape, learnable = False)

    def calc_maccs(self):
        self.maccs = self.in_shape * self.out_shape

    def _calc_flops(self):
        bias = self.params['bias'].value
        if bias:
            self.flops = 2 * self.in_shape * self.out_shape
        else:
            self.flops = (2 * self.in_shape - 1) * self.out_shape

    def _create_args(self):
        self.args = [self.in_shape, self.out_shape]
        self.kwargs = {'bias' : self.params['bias'].value}

        
class Conv2dTmpl(ModuleTemplate_): 
    def __init__(self, in_shape = None, 
                 out_channels = 32, 
                 kernel_size = (7, 7), 
                 stride = (1, 1), 
                 padding = False, 
                 padding_size = (0, 0), 
                 padding_mode = 'zeros',
                 dilation = (1, 1), 
                 groups = 1,
                 bias = True):
               
        super().__init__(in_shape = in_shape, 
                        out_channels = out_channels, 
                        kernel_size = kernel_size, 
                        stride = stride, 
                        padding = padding,
                        padding_size = padding_size, 
                        padding_mode = padding_mode,
                        dilation = dilation, 
                        groups = groups, 
                        bias = bias)
        # utils.print_nonprivate_properties(self)
        self.module_name = 'Conv2d'
        self._calc_padding_size()
        # self._postfix_conv_params()
        self.calc_out_shape()

    def gen_params(self):
        super().gen_params()
        self._calc_padding_size()
        # self._postfix_conv_params()        

    def _init_param_values_list(self):
        self.param_values = {}
        self.param_values['out_channels'] = [8, 16, 32, 64, 128, 256]
        self.param_values['kernel_size'] = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]
        self.param_values['stride'] = [(1, 1), (2, 2), (3, 3), (4, 4)]
        self.param_values['padding'] = [True, False]
        self.params['dilation'].learnable = False
        self.params['groups'].learnable = False
        self.params['bias'].learnable = False
        self.params['padding_mode'].learnable = False

    def _calc_padding_size(self):
        padding = self.params['padding'].value
        kernel_size = self.params['kernel_size'].value
        if padding:
            self.params['padding_size'] = ParameterEx((kernel_size[0] // 2, kernel_size[1] // 2), learnable=False)
        else:
            self.params['padding_size'] = ParameterEx((0, 0), learnable=False)

    def calc_out_shape(self):
        if self.in_shape is None:
            self.out_shape = None
        else:
            in_shape = self.in_shape
            padding = self.params['padding_size'].value
            dilation = self.params['dilation'].value
            kernel_size = self.params['kernel_size'].value
            stride = self.params['stride'].value

            out_shape_height = (in_shape[1] + 2*padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
            out_shape_width = (in_shape[2] + 2*padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

            self.out_shape = (self.params['out_channels'].value, out_shape_height, out_shape_width)          

    def calc_maccs(self):
        self.calc_out_shape()
        if not self.out_shape:
            kernel = self.params['kernel_size'].value
            self.maccs = kernel[0] * kernel[1] * self.in_shape[0] * \
                self.out_shape[0] * self.out_shape[1] * self.out_shape[2]
        else:
            self.maccs = 0

    def _calc_flops(self):
        kernel = self.params['kernel_size'].value
        bias = self.params['bias'].value
        if bias:
            self.flops = 2 * kernel[0] * kernel[1] * self.in_shape[0] * \
                self.out_shape[0] * self.out_shape[1] * self.out_shape[2]
        else:
            self.flops = (2 * kernel[0] * kernel[1] - 1) * self.in_shape[0] * \
                self.out_shape[0] * self.out_shape[1] * self.out_shape[2]
        
    def _create_args(self):
        self.args = [self.in_shape[0],
                     self.out_shape[0],
                     self.params['kernel_size'].value]
        self.kwargs = {'stride' : self.params['stride'].value,
                       'padding' : self.params['padding_size'].value,
                       'dilation' : self.params['dilation'].value,
                       'groups' : self.params['groups'].value,
                       'bias' : self.params['bias'].value,
                       'padding_mode' : self.params['padding_mode'].value,
                      }

    def __repr__(self):
        kernel_size = str(self.params['kernel_size'].value)
        stride = str(self.params['stride'].value)        
        repr_str = '{:12} {:14} {:8} {:8}'.format(self.module_name, 
                                                  str(self.out_shape), 
                                                  kernel_size,
                                                  stride)
        return repr_str        


class MaxPool2dTmpl(ModuleTemplate_):
    def __init__(self, in_shape = None,
                 kernel_size = (2, 2),
                 stride = (2, 2),           
                 padding = False,
                 padding_size = (0, 0),
                 dilation = (1, 1),
                 return_indices = False,
                 ceil_mode=False):
        super().__init__(in_shape = in_shape,
                         kernel_size = kernel_size,
                         stride = stride,
                         padding = padding,
                         padding_size = padding_size,
                         dilation = dilation,
                         return_indices = return_indices,
                         ceil_mode = ceil_mode)
        
        self.module_name = 'MaxPool2d'
        self._calc_padding_size()
        # self._postfix_conv_params()
        self.calc_out_shape()
    
    def gen_params(self):
        super().gen_params()
        self._calc_padding_size()
        # self._postfix_conv_params()        

    def _calc_padding_size(self):
        padding = self.params['padding'].value
        kernel_size = self.params['kernel_size'].value
        if padding:
            self.params['padding_size'] = ParameterEx((kernel_size[0] // 2, kernel_size[1] // 2), learnable=False)
        else:
            self.params['padding_size'] = ParameterEx((0, 0), learnable=False)

    def _init_param_values_list(self):
        self.param_values = {}
        self.param_values['kernel_size'] = [(2, 2), (3, 3), (4, 4)]
        self.param_values['stride'] = [(1, 1), (2, 2)]
        self.param_values['padding'] = [True, False]
        self.params['dilation'].learnable = False
        self.params['return_indices'].learnable = False
        # ceil_mode doesn't seem to work correctly
        self.params['ceil_mode'].learnable = False

    # def _postfix_conv_params(self):
        # padding = self.params['padding'].value
        # kernel_size = self.params['kernel_size'].value
        # if padding:
        #     self.params['padding_size'] = ParameterEx(kernel_size // 2, learnable=False)
        # else:
        #     self.params['padding_size'] = ParameterEx(0, False)
        # for param_name in ['kernel_size', 'stride', 'padding_size', 'dilation']:
        #     self._int2tuple_broadcast(param_name)

    def calc_out_shape(self):
        if self.in_shape is None:
            self.out_shape = None
        else:
            in_shape = self.in_shape
            padding = self.params['padding_size'].value
            dilation = self.params['dilation'].value
            kernel_size = self.params['kernel_size'].value
            stride = self.params['stride'].value
            ceil_mode = self.params['ceil_mode'].value

            out_shape_height = (in_shape[1] + 2*padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
            out_shape_width = (in_shape[2] + 2*padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
            if ceil_mode:
                out_shape_height = ceil(out_shape_height)
                out_shape_width = ceil(out_shape_width)
            else:
                out_shape_height = floor(out_shape_height)
                out_shape_width = floor(out_shape_width)

            self.out_shape = (in_shape[0], out_shape_height, out_shape_width)

    def _create_args(self):
        self.args = [self.params['kernel_size'].value]
        self.kwargs = {'stride' : self.params['stride'].value,
                       'padding' : self.params['padding_size'].value,                       
                       'dilation' : self.params['dilation'].value,                      
                       'ceil_mode' : self.params['ceil_mode'].value,
                      }

    def __repr__(self):
        kernel_size = str(self.params['kernel_size'].value)
        stride = str(self.params['stride'].value)        
        repr_str = '{:12} {:14} {:8} {:8}'.format(self.module_name, 
                                                  str(self.out_shape), 
                                                  kernel_size,
                                                  stride)
        return repr_str        


class AvgPool2dTmpl(ModuleTemplate_):
    def __init__(self, in_shape = None,
                 kernel_size = (2, 2),
                 stride = (2, 2),           
                 padding = False,
                 padding_size = (0, 0),
                 ceil_mode=False):
        super().__init__(in_shape = in_shape,
                         kernel_size = kernel_size,
                         stride = stride,
                         padding = padding,
                         padding_size = padding_size,
                         ceil_mode = ceil_mode)
        
        self.module_name = 'AvgPool2d'
        self._calc_padding_size()
        # self._postfix_conv_params()
        self.calc_out_shape()
    
    def gen_params(self):
        super().gen_params()
        self._calc_padding_size()
        # self._postfix_conv_params()        

    def _calc_padding_size(self):
        padding = self.params['padding'].value
        kernel_size = self.params['kernel_size'].value
        if padding:
            self.params['padding_size'] = ParameterEx((kernel_size[0] // 2, kernel_size[1] // 2), learnable=False)
        else:
            self.params['padding_size'] = ParameterEx((0, 0), learnable=False)

    def _init_param_values_list(self):
        self.param_values = {}
        self.param_values['kernel_size'] = [(2, 2), (3, 3), (4, 4)]
        self.param_values['stride'] = [(1, 1), (2, 2)]
        self.param_values['padding'] = [True, False]
        # ceil_mode doesn't seem to work correctly
        self.params['ceil_mode'].learnable = False

    def calc_out_shape(self):
        if self.in_shape is None:
            self.out_shape = None
        else:
            in_shape = self.in_shape
            padding = self.params['padding_size'].value
            kernel_size = self.params['kernel_size'].value
            stride = self.params['stride'].value
            ceil_mode = self.params['ceil_mode'].value

            out_shape_height = (in_shape[1] + 2*padding[0] - kernel_size[0]) / stride[0] + 1
            out_shape_width = (in_shape[2] + 2*padding[1] - kernel_size[1]) / stride[1] + 1
            if ceil_mode:
                out_shape_height = ceil(out_shape_height)
                out_shape_width = ceil(out_shape_width)
            else:
                out_shape_height = floor(out_shape_height)
                out_shape_width = floor(out_shape_width)

            self.out_shape = (in_shape[0], out_shape_height, out_shape_width)

    def _create_args(self):
        self.args = [self.params['kernel_size'].value]
        self.kwargs = {'stride' : self.params['stride'].value,
                       'padding' : self.params['padding_size'].value,
                       'ceil_mode' : self.params['ceil_mode'].value,
                      }

    def __repr__(self):
        kernel_size = str(self.params['kernel_size'].value)
        stride = str(self.params['stride'].value)        
        repr_str = '{:12} {:14} {:8} {:8}'.format(self.module_name, 
                                                  str(self.out_shape), 
                                                  kernel_size,
                                                  stride)
        return repr_str        


class GlobalAvgPool2dTmpl(ModuleTemplate_):
    def __init__(self, in_shape = None):
        super().__init__(in_shape = in_shape)
        
        self.module_name = 'AvgPool2d'
        self.calc_out_shape()
    
    def gen_params(self):
        super().gen_params()

    def _init_param_values_list(self):
        self.param_values = {}

    def calc_out_shape(self):
        if self.in_shape is None:
            self.out_shape = None
        else:
            in_shape = self.in_shape
            self.out_shape = (in_shape[0], 1, 1)    # out_shape is (num_channels, 1, 1)

    def _create_args(self):
        self.args = [(self.in_shape[1], self.in_shape[2])]    # kernel_size = in_shape for AvgPool2d
        self.kwargs = {}

    def __repr__(self):
        repr_str = '{:12} {:14}'.format('Global' + self.module_name, str(self.out_shape))
        return repr_str        
    

class BatchNorm2dTmpl(ModuleTemplate_):
    def __init__(self, in_shape = None, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__(in_shape = in_shape, eps = eps, momentum = momentum, affine = affine, 
                        track_running_stats = track_running_stats)                
        self.module_name = 'BatchNorm2d'
        self.calc_out_shape()

    def _init_param_values_list(self):
        self.param_values = {}
        self.params['eps'].learnable = False
        self.params['momentum'].learnable = False
        self.params['affine'].learnable = False
        self.params['track_running_stats'].learnable = False

    def _create_args(self):
        self.args = [self.in_shape[0]]
        self.kwargs = {
            'eps' : self.params['eps'].value,
            'momentum' : self.params['momentum'].value,
            'affine' : self.params['affine'].value,
            'track_running_stats' : self.params['track_running_stats'].value
        }        

        
class ReLUTmpl(ModuleTemplate_):
    def __init__(self, in_shape = None, inplace = True):
        super().__init__(in_shape=in_shape)
        
        self.module_name = 'ReLU'
        self.calc_out_shape()

    # def _init_param_values_list(self):
    #     self.params['inplace'].learnable = False

    # def _create_args(self):
    #     self.args = []
    #     self.kwargs = {
    #         'inplace' : self.params['inplace'].value
    #     }


class LeakyReLUTmpl(ModuleTemplate_):
    def __init__(self, in_shape = None, negative_slope = 0.01, inplace=True):
        super().__init__(in_shape = in_shape, 
                        negative_slope = negative_slope,
                        inplace = inplace)
        self.module_name = 'LeakyReLU'
        self.calc_out_shape()
    
    def _init_param_values_list(self):
        self.param_values = {}
        self.param_values['negative_slope'] = [0.1, 0.01, 0.001]
        self.params['inplace'].learnable = False
    
    def _create_args(self):
        self.args = []
        self.kwargs = {
            'negative_slope' : self.params['negative_slope'].value,
            'inplace' : self.params['inplace'].value
        }
    def __repr__(self):
        negative_slope = self.params['negative_slope'].value
        repr_str = '{:12} {:14} {:8}'.format( self.module_name,
                                        str(self.out_shape),
                                        '(' + str(negative_slope) + ')' )
        return repr_str


class Dropout2dTmpl(ModuleTemplate_):
    def __init__(self, in_shape = None, p = 0.7, inplace = False):
        super().__init__(in_shape = in_shape, p = p)
        self.module_name = 'Dropout2d'
        self.calc_out_shape()           

    def _init_param_values_list(self):
        self.param_values = {}
        self.param_values['p'] = np.linspace(0.05, 0.95, num = 19)

    def _create_args(self):
        self.args = [self.params['p'].value]
        self.kwargs = {}

    def __repr__(self):
        p = round(self.params['p'].value, 2)
        repr_str = '{:12} {:14} {:8}'.format( self.module_name,
                                        str(self.out_shape),
                                        '(' + str(p) + ')' )
        return repr_str


class FlattenTmpl(ModuleTemplate_):
    def __init__(self, in_shape = None):        
        super().__init__(in_shape = in_shape)
        self.module_name = 'Flatten'
        self.calc_out_shape()

    def calc_out_shape(self):
        if self.in_shape is None:
            self.out_shape = None
        else:
            self.out_shape = 1
            # print(self.in_shape, type(self.in_shape))
            for dim in self.in_shape:
                self.out_shape *= dim
    
    def instantiate_module(self):
        self._create_args()
        klass = globals()[self.module_name]
        instance = klass(*self.args, **self.kwargs)
        return instance

######################### EX - SEQUENTIAL ########################
    
class SequentialEx(nn.Sequential):    
    def __init__(self, in_shape, out_shape, *templates):
        self.in_shape = in_shape
        self.module_templates = list(templates)
        self._set_template_shapes(in_shape, out_shape)
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

            # mod_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
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
    
    def _set_template_shapes(self, in_shape, out_shape):        
        for i, template in enumerate(self.module_templates):                        
            # first elem
            if i == 0:
                template.set_in_shape(in_shape)
                template.calc_out_shape()
            # last elem
            elif i == len(self.module_templates) - 1:
                template.set_in_shape(previous_out_shape)
                template.set_out_shape(out_shape)
            # middle elem
            else:
                template.set_in_shape(previous_out_shape)
                template.calc_out_shape()            
            previous_out_shape = template.out_shape
            self.module_templates[i] = template
            
    def _instantiate_modules(self):
        modules_instances = []
        for template in self.module_templates:             
            # debug info
            print(template.module_name, 'in_shape:', template.in_shape, 'out_shape:', template.out_shape)
            module = template.instantiate_module()
            module.in_shape = template.in_shape
            modules_instances.append(module)
        self.modules_instances = modules_instances

################ LAYER TEMPLATES ####################

class LayerTemplate():
    '''
        Create Layer - main template followed by auxilaries:
            Ex. Conv - DropOut - MaxPool - ReLU
        Two ways of creation:
         - From preset of templates passed as args: 
         
            conv_layer = ConvLayerTmpl((3, 32, 32), tmpl1, tmpl2, tmpl3)
         
         - Randomly generated  main template followed by auxilaries:

            conv_layer = ConvLayerTmpl((3, 32, 32))
            conv_layer.gen_rand_layer()

        The list of aux'es and probabilities of their generation must be listed in self.freq_dict.
        All template parameters are generated randomly from internal lists of each
        template.
        
        Upon creation only needs 'in_shape'. 'out_shape' is calculated (upon generation) or set from 
        last template provided. If generated from templates - their shapes are synced:
        layer.in_shape -> tmpl1.in_shape -> tmpl2.in_shape = tmpl1.out_shape -> ... -> layer.out_shape = last_tmpl.out_shape

        If one of aux or main templates rendered zero in out_shape (ex. (32, 0, 0)), the flag
        self.zero_shape is set and further template generation stops.
    '''

    def __init__(self, in_shape, *templates):
        self.templates = list(templates)
        self.in_shape = in_shape
        self.out_shape = in_shape
        self.zero_shape = False

        # if templates:
        #     self._sync_shapes()            

    def _sync_shapes(self):        
        for tmpl in self.templates:
            tmpl.set_in_shape(self.out_shape)
            tmpl.calc_out_shape()
            zero_shape = self._check_zero_shape(tmpl)
            if zero_shape:
                print(tmpl.module_name, tmpl.in_shape, tmpl.out_shape)
                self.zero_shape = True
            #     self.out_shape = next_in_shape
            # else:
            self.out_shape = tmpl.out_shape
        
    def set_in_shape(self, in_shape):
        self.in_shape = in_shape
        self.out_shape = in_shape
        self._sync_shapes()

    def gen_rand_layer(self):
        self.templates = []
        self._generate_main_template()        
        self._generate_aux_templates()

    def _generate_main_template(self):
        '''
        Children should overwrite this method
        '''
        pass

    def _generate_aux_templates(self):
        freq_dict = utils.shuffle_dict(self.freq_dict)    # children will override this
        for tmpl_name in freq_dict:
            tmpl_freq = freq_dict[tmpl_name]
            if tmpl_freq > random.random():                
                self._generate_template(tmpl_name, in_shape = self.out_shape)
                if self.zero_shape: break
       
    def _generate_template(self, tmpl_name, in_shape):
        # appends template and updates self.out_shape
        tmpl = self._instantiate_template(tmpl_name, in_shape = in_shape)
        tmpl.gen_params()
        tmpl.calc_out_shape()
        zero_shape = self._check_zero_shape(tmpl)
        # print(tmpl.module_name, tmpl.in_shape, tmpl.out_shape)

        if zero_shape:
            # print(tmpl.module_name + ': ZERO SHAPE', tmpl.out_shape)
            self.zero_shape = True
            self.out_shape = in_shape
        else:
            self.templates.append(tmpl)
            self.out_shape = tmpl.out_shape
    
    def _check_zero_shape(self, tmpl):
        '''
        Overwriten by children 1d or 2d
        '''
        return False

    def _instantiate_template(self, tmpl_name, **kwargs):
        klass = globals()[tmpl_name]
        template = klass(**kwargs)
        return template

    def __repr__(self):
        out_str = ''
        for tmpl in self.templates:
            out_str += repr(tmpl) + '\n'            
        return out_str        


class ConvLayerTmpl(LayerTemplate):
    '''
    Starts with main Conv2dTmpl template followed by combination of 
    auxilary templates from self.freq_dict
    '''
    def __init__(self, in_shape, *templates):        
        self.freq_dict = {'ReLUTmpl' : 1,       
                        'MaxPool2dTmpl': 0.5,
                        'BatchNorm2dTmpl': 0.5,
                        'Dropout2dTmpl': 0}

        super().__init__(in_shape, *templates)
            
    def _generate_main_template(self):
        self._generate_template('Conv2dTmpl', in_shape = self.in_shape)        

    def _check_zero_shape(self, tmpl):
        '''
        Shape check for 2d layers
        '''
        zero_shape = (tmpl.out_shape[1] < 1) or (tmpl.out_shape[2] < 1)
        return zero_shape


class LinLayerTmpl(LayerTemplate):
    def __init__(self, in_shape, *templates):
        self.freq_dict = {'ReLUTmpl' : 1,
                          'Dropout2dTmpl': 0.5}
        super().__init__(in_shape, *templates)

    def _generate_main_template(self):
        self._generate_template('LinearTmpl', in_shape = self.in_shape)        

    def _check_zero_shape(self, tmpl):
        '''
        Shape check for 1d layers
        '''
        zero_shape = tmpl.out_shape < 1
        return zero_shape


class GapLayerTmpl(LayerTemplate):
    '''
        Global Average Pool layer. Single template
    '''
    def __init__(self, in_shape, *templates):
        super().__init__(in_shape, *templates)
    
    def gen_rand_layer(self):
        self._generate_main_template()

    def _generate_main_template(self):
        self._generate_template('GlobalAvgPool2dTmpl', in_shape = self.in_shape)


class FlatLayerTmpl(LayerTemplate):
    def __init__(self, in_shape, *templates):
        super().__init__(in_shape, *templates)
    
    def gen_rand_layer(self):
        self._generate_main_template()

    def _generate_main_template(self):
        self._generate_template('FlattenTmpl', in_shape = self.in_shape)


####################### MODEL TEMPLATE ##################

class ModelTmpl():
    '''
        Model created from layers
    '''
    def __init__(self, in_shape = None, out_shape = None, *layers):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.layers = list(layers)
        if layers:
            self._sync_shapes()

    def _sync_shapes(self):
        next_in_shape = self.in_shape
        for layer in self.layers:
            layer.set_in_shape(next_in_shape)
            if layer.zero_shape:
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
        self._sync_shapes()

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


class ModelTmpl_():
    '''
        ModelTmpl created from templates
    '''
    def __init__(self, in_shape = None, out_shape = None, *templates):        
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.zero_shape = False
        self.templates = list(templates)

        if templates:
            self._sync_shapes()

    def _sync_shapes(self):    
        next_in_shape = self.in_shape
        for tmpl in self.templates:
            tmpl.set_in_shape(next_in_shape)
            tmpl.calc_out_shape()
            is_zero_shape = tmpl.check_zero_shape()
            if is_zero_shape:
                print(tmpl.module_name, tmpl.in_shape, tmpl.out_shape)
                self.templates.remove(tmpl)
            else:
                next_in_shape = tmpl.out_shape
        self._set_last_shape()
    
    def _set_last_shape(self):
        self.templates[-1].set_out_shape(self.out_shape)        
        
    def set_shapes(self, in_shape, out_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self._sync_shapes()
    
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

