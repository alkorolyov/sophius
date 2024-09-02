import sophius
import torch
import torch.nn as nn
import random
import pytest
import numpy as np
from sophius.templates import *
# pylint: skip-file

class TestTemplate():
    config_data = {}


class TestModuleTmpl(ModuleTemplate_):
    config_data = {
        'option_1': {
            'default': Parameter(1),
            'range': [-1, 0, 1]
        },
        'option_2': {
            'default': Parameter(False, learnable=False)
        }
    }


class TestConfig(ModuleTemplate_):
    config_data = {
        'option_1': {
            'default': Parameter(1),
            'range': [-1, 0, 1]
        }
    }
    def _update_out_shape(self):
        self.out_shape = self.config['option_1'].value


def test_config_get_defaults():
    template = TestTemplate()
    template.config_data = {
        'option_1': {
            'default': Parameter(1),
            'range': [-1, 0, 1]
        },
        'option_2': {
            'default': Parameter(False, learnable=False)
        }
    }
    config = ConfigGenerator(template).get()
    assert config['option_1'].value == 1
    assert config['option_1'].learnable == True
    assert config['option_2'].value == False    
    assert config['option_2'].learnable == False

def test_config_get_args():
    template = TestTemplate()
    template.config_data = {
        'option_1': {
            'default': Parameter(1, learnable=False),
            'range': [-1, 0, 1]
        }
    }
    config = ConfigGenerator(template).get(option_1=2)
    assert config['option_1'].value == 2
    assert config['option_1'].learnable == False

def test_config_get_extra_args_exception():
    template = TestTemplate()
    with pytest.raises(ValueError):
        assert ConfigGenerator(template).get(option_1=2)

def test_config_get_random():
    template = TestTemplate()
    template.config_data = {
        'option_1': {
            'default': Parameter('string', learnable=True)
        },
        'option_2': {
            'defalt': Parameter(1.25, learnable=False),
            'range': [-1.25, 0, 1.25]
        }
    }
    random.seed(0)
    config = ConfigGenerator(template).get_random()
    # print(config)
    assert config['option_1'].value == 'string'
    assert config['option_1'].learnable == True
    assert config['option_2'].value == 0
    assert config['option_2'].learnable == True

def test_moduletmpl_is_zero_tuple():
    template = TestModuleTmpl(in_shape = (0, 1))
    assert template.is_zero_shape == True

def test_moduletmpl_is_zero_single():
    template = TestModuleTmpl(in_shape = 1)
    assert template.is_zero_shape == False

def test_moduletmpl_is_zero_None():
    template = TestModuleTmpl(in_shape = None)
    assert template.is_zero_shape == False

def test_moduletmpl_in_shape_change():
    template = TestModuleTmpl(in_shape = None)
    template.in_shape = (1, 1)
    assert template.out_shape == (1, 1)
    assert template.is_zero_shape == False

def test_moduletmpl_config_change():        
    template = TestConfig(in_shape = 0, option_1 = (1, 0))
    template.config = ConfigGenerator(template).get()
    assert template.out_shape == (1)
    assert template.is_zero_shape == False

def test_moduletmpl_config_value_change():
    template = TestConfig(in_shape = 0, option_1 = (1, 0))
    assert template.config['option_1'].value == (1, 0)
    template.config['option_1'].value = (1, 1)
    template.update_shape()
    assert template.out_shape == (1, 1)
    assert template.is_zero_shape == False

def test_moduletmpl_config_change_key():
    template = TestConfig(in_shape = 0, option_1 = (1, 0))
    template.config['option_1'] = Parameter((1, 1))
    template.update_shape()
    assert template.out_shape == (1, 1)
    assert template.is_zero_shape == False

def test_convtmpl_padding_size():
    template = ConvTemplate_(in_shape=(1, 2, 2), padding=True, kernel_size=(5, 5))
    assert template._calc_padding_size() == (2, 2)

def test_convtmpl_padding_size_zero():
    template = ConvTemplate_(in_shape=(1, 2, 2), padding=False, kernel_size=(5, 5))
    assert template._calc_padding_size() == (0, 0)

def test_convtmpl_out_shape():
    template = ConvTemplate_(in_shape = (1, 4, 4))
    template.config = {
        'out_channels': Parameter(32),
        'kernel_size': Parameter((3, 3)),
        'stride': Parameter((1, 1)),
        'padding': Parameter(True),
        'dilation': Parameter((1, 1)),
        'ceil_mode': Parameter(False)
    }
    assert template.out_shape == (32, 4, 4)
    template.config['out_channels'].value = 16
    template.update_shape()
    assert template.out_shape == (16, 4, 4)
    template.config['padding'] = Parameter(False)
    template.update_shape()
    assert template.out_shape == (16, 2, 2)
    template.config['padding'] = Parameter(True)
    template.config['stride'] = Parameter((2, 2))
    template.update_shape()
    assert template.out_shape == (16, 2, 2)
    template.config['stride'].value = (5, 5)
    template.update_shape()
    assert template.out_shape == (16, 1, 1)

def test_convtmpl_rand():
    random.seed(0)
    template = ConvTemplate_()
    template.config_data = {
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
            'default': Parameter( (1, 1) )
        },
        'ceil_mode': {
            'default': Parameter(False)
        }
    }
    template.gen_rand_config()
    assert template.config['out_channels'].value == 4
    assert template.config['kernel_size'].value == (2, 2)
    assert template.config['stride'].value == (1, 1)
    assert template.config['padding'].value == False

def test_lintmpl():
    tmpl = LinearTmpl(10, out_features = 32, bias = False)
    assert tmpl.in_shape == 10
    assert tmpl.out_shape == 32
    assert tmpl.config['bias'].value is False    
    random.seed(0)
    tmpl.gen_rand_config()
    assert tmpl.out_shape == tmpl.config['out_features'].value
    assert tmpl.out_shape == 2048
    module = tmpl.instantiate_module()
    batch_size = 4
    input = torch.randn((batch_size, 10))
    output = module(input.float())
    assert tmpl.out_shape == (output.shape[1])

def test_batch_norm_tmpl():
    tmpl = BatchNorm2dTmpl(in_shape=(1, 4, 4))
    module = tmpl.instantiate_module()
    batch_size = 4
    input = torch.randn((batch_size, 1, 4, 4))
    output = module(input.float())
    assert tmpl.out_shape == (output.shape[1], output.shape[2], output.shape[3])

def test_relu_tmpl():
    tmpl = ReLUTmpl()
    relu = tmpl.instantiate_module()
    input = torch.tensor([[ -1, 0],
                        [0.1, 1]], dtype=torch.float32)
    output = relu(input)
    result = torch.tensor([[  0, 0],
                        [0.1, 1]], dtype=torch.float32)
    dist = torch.dist(output, result)
    # print(dist)
    assert dist == torch.tensor(0)

def test_leakyrelu_tmpl():
    tmpl = LeakyReLUTmpl()
    relu = tmpl.instantiate_module()
    input = torch.tensor([[ -1, 0],
                        [0.1, 1]], dtype=torch.float32)
    output = relu(input)
    result = torch.tensor([[  0, 0],
                        [0.1, 1]], dtype=torch.float32)
    dist = torch.dist(output, result)
    # print(dist)
    assert dist <= torch.tensor(0.2)

def test_prelu_tmpl():
    tmpl = PReLUTmpl(in_shape=(2, 4, 4), all_channels=True)
    module = tmpl.instantiate_module()
    assert tmpl._kwargs['num_parameters'] == 2
    batch_size = 4
    input = torch.randn((batch_size, 2, 4, 4))
    output = module(input.float())
    assert tmpl.out_shape == (output.shape[1], output.shape[2], output.shape[3])

def test_dropout_tmpl():
    tmpl = Dropout2dTmpl(in_shape=(4, 4))
    module = tmpl.instantiate_module()
    batch_size = 4
    input = torch.randn((batch_size, 4, 4))
    output = module(input.float())
    assert tmpl.out_shape == (output.shape[1], output.shape[2])

def test_flatten_tmpl():
    tmpl = FlattenTmpl(in_shape=(3, 4, 4))
    module = tmpl.instantiate_module()
    batch_size = 4
    input = torch.randn((batch_size, 3, 4, 4))
    output = module(input.float())
    assert tmpl.out_shape == (output.shape[1])

def test_conv_tmpl():    
    tmpl = Conv2dTmpl(in_shape=(1, 4, 4))
    module = tmpl.instantiate_module()
    batch_size = 4
    input = torch.randn((batch_size, 1, 4, 4))
    output = module(input.float())
    assert tmpl.out_shape == (output.shape[1], output.shape[2], output.shape[3])

def test_maxpool_tmpl():
    tmpl = MaxPool2dTmpl(in_shape=(1, 4, 4))
    module = tmpl.instantiate_module()
    batch_size = 4
    input = torch.randn((batch_size, 1, 4, 4))
    output = module(input.float())
    assert tmpl.out_shape == (output.shape[1], output.shape[2], output.shape[3])

def test_avgpool_tmpl():
    tmpl = AvgPool2dTmpl(in_shape=(1, 4, 4))
    module = tmpl.instantiate_module()
    batch_size = 4
    input = torch.randn((batch_size, 1, 4, 4))
    output = module(input.float())
    assert tmpl.out_shape == (output.shape[1], output.shape[2], output.shape[3])

def test_gap_tmpl():
    tmpl = GlobalAvgPool2dTmpl(in_shape=(1, 4, 4))
    module = tmpl.instantiate_module()
    batch_size = 4
    input = torch.randn((batch_size, 1, 4, 4))
    output = module(input.float())
    assert tmpl.out_shape == (output.shape[1], output.shape[2], output.shape[3])
    tmpl.gen_rand_config()
    module = tmpl.instantiate_module()
    batch_size = 4
    input = torch.randn((batch_size, 1, 4, 4))
    output = module(input.float())
    assert tmpl.out_shape == (output.shape[1], output.shape[2], output.shape[3])

def test_seq_tmpl():
    conv = Conv2dTmpl((3, 4, 4), kernel_size=(3, 3), stride=(2, 2))
    flat = FlattenTmpl()
    gap = GlobalAvgPool2dTmpl()
    lin = LinearTmpl()
    seq = SequentialEx((3, 4, 4), 10, conv, gap, flat, lin)
    batch_size = 4
    input = torch.randn((batch_size, 3, 4, 4))
    output = seq(input.float())
    assert (10) == (output.shape[1])

def test_layertmpl_init():
    conv = Conv2dTmpl(out_channels=16, kernel_size=(3, 3), stride=(2, 2))
    flat = FlattenTmpl()
    gap = GlobalAvgPool2dTmpl()
    layer = LayerTemplate_(None, conv, gap, flat)
    assert layer.in_shape == None
    assert layer.out_shape == None
    layer.in_shape = (3, 4, 4)
    assert layer.out_shape == (16)
    layer.templates[0].config['out_channels'].value = 32
    layer.sync_shapes()
    assert layer.out_shape == (32)

def test_layertmpl_instantiate():
    conv = Conv2dTmpl(out_channels=16, kernel_size=(3, 3), stride=(2, 2))
    flat = FlattenTmpl()
    gap = GlobalAvgPool2dTmpl()
    layer = LayerTemplate_((3, 4, 4), conv, gap, flat)
    layer_instance = layer.instantiate_layer()
    input = torch.randn((32, 3, 4, 4))
    output = layer_instance(input.float())    
    assert layer.out_shape == output.shape[1]

def test_layertmpl_random():    
    layer = LayerTemplate_()
    random.seed(0)
    layer.freq_dict = {
        'activation': ['ReLUTmpl'],
        'MaxPool2dTmpl': 1,
        'BatchNorm2dTmpl': 1,
        'AvgPool2dTmpl': 1,
        'Dropout2dTmpl': 1
    }
    layer.gen_rand_layer()
    assert layer.out_shape == None
    layer.in_shape = (1, 5, 5)
    assert layer.out_shape == (1, 2, 2)

def test_convlayer_tmpl():
    random.seed(0)
    layer = ConvLayerTmpl()
    layer.freq_dict = {
        'activation' : ['ReLUTmpl', 'LeakyReLUTmpl', 'PReLUTmpl'],
        'MaxPool2dTmpl': 0.5,
        'BatchNorm2dTmpl': 0.5,
        'AvgPool2dTmpl': 0.5,
        'Dropout2dTmpl': 0
    }
    layer.gen_rand_layer()    
    assert layer.out_shape is None
    random.seed(0)
    layer = ConvLayerTmpl((3, 4, 4))
    layer.freq_dict = {
        'activation' : ['ReLUTmpl', 'LeakyReLUTmpl', 'PReLUTmpl'],
        'MaxPool2dTmpl': 0.5,
        'BatchNorm2dTmpl': 0.5,
        'AvgPool2dTmpl': 0.5,
        'Dropout2dTmpl': 0
    }
    layer.gen_rand_layer()
    assert layer.out_shape == (192, 1, 1)
    random.seed(0)
    layer = ConvLayerTmpl((3, 4, 4))
    layer.freq_dict = {
        'activation' : ['ReLUTmpl', 'LeakyReLUTmpl', 'PReLUTmpl'],
        'MaxPool2dTmpl': 0.5,
        'BatchNorm2dTmpl': 0.5,
        'AvgPool2dTmpl': 0.5,
        'Dropout2dTmpl': 0
    }
    layer.gen_rand_layer()
    layer_instance = layer.instantiate_layer()
    input = torch.randn((32, 3, 4, 4))
    output = layer_instance(input.float())    
    assert layer.out_shape == (output.shape[1], output.shape[2], output.shape[3])

def test_linlayer_tmpl():
    random.seed(0)
    layer = LinLayerTmpl()
    layer.freq_dict = {
        'activation' : ['ReLUTmpl', 'LeakyReLUTmpl', 'PReLUTmpl'],
        'Dropout2dTmpl': 0.5
    }    
    layer.gen_rand_layer()
    assert layer.out_shape is None
    random.seed(0)
    layer = LinLayerTmpl(32)
    layer.freq_dict = {
        'activation' : ['ReLUTmpl', 'LeakyReLUTmpl', 'PReLUTmpl'],
        'Dropout2dTmpl': 0.5
    }    
    layer.gen_rand_layer()
    assert layer.out_shape == 2048

def test_gaplayer_tmpl():    
    random.seed(0)
    layer = GapLayerTmpl()
    layer.gen_rand_layer()
    assert layer.out_shape is None
    random.seed(0)
    layer = GapLayerTmpl((3, 4, 4))
    layer.gen_rand_layer()
    layer_instance = layer.instantiate_layer()
    input = torch.randn((32, 3, 4, 4))
    output = layer_instance(input.float())    
    assert layer.out_shape == (3, 1, 1)
    assert layer.out_shape == (output.shape[1], 1, 1)

def test_model_templates_init_empty():
    model = ModelTmpl_()
    model.sync_shapes()
    assert model.out_shape is None

def test_model_templates_init_inshape():
    model = ModelTmpl_((3, 4, 4))
    model.sync_shapes()
    assert model.out_shape is None

def test_model_templates_init_templaes():
    conv = Conv2dTmpl()
    model = ModelTmpl_(None, None, conv)
    model.sync_shapes()
    assert model.out_shape is None

def test_model_templates_init():
    conv = Conv2dTmpl()
    flat = FlattenTmpl()
    lin = LinearTmpl()
    model = ModelTmpl_((3, 4, 4), 10, conv, flat, lin)
    model.sync_shapes()
    assert model.out_shape is 10
    model_instance = model.instantiate_model()
    input = torch.randn((32, 3, 4, 4))
    output = model_instance(input.float())
    assert model.out_shape == 10
    assert model.out_shape == output.shape[1]

def test_model_layers_init_inshape():
    model = ModelTmpl((3, 4, 4))
    model.sync_shapes()
    assert model.out_shape is None

def test_model_layers_init_templaes():
    conv = ConvLayerTmpl()
    model = ModelTmpl(None, None, conv)
    model.sync_shapes()
    assert model.out_shape is None

def test_model_layers_init():
    conv = ConvLayerTmpl()
    conv.gen_rand_layer()
    flat = FlatLayerTmpl()
    flat.gen_rand_layer()
    lin = LastLinLayerTmpl()
    lin.gen_rand_layer()
    model = ModelTmpl((3, 4, 4), 10, conv, flat, lin)
    model.sync_shapes()
    assert model.out_shape is 10
    model_instance = model.instantiate_model()
    input = torch.randn((32, 3, 4, 4))
    output = model_instance(input.float())
    assert model.out_shape == 10
    assert model.out_shape == output.shape[1]    