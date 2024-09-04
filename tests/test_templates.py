import sophius
import torch
import torch.nn as nn
import random
import pytest
import numpy as np
from sophius.templates import *
# pylint: skip-file

class TestTemplate(ModuleTemplate_):
    config = {}


class MockTemplate(ModuleTemplate_):
    config = {
        'option_1': {
            'default': 1,
            'range': [-1, 0, 1]
        },
        'option_2': {
            'default': False,
        }
    }


def test_config_get_defaults():
    template = MockTemplate()
    assert template.params['option_1'] == 1
    assert not template.params['option_2']


def test_config_get_random():
    template = MockTemplate()
    random.seed(0)
    template.gen_rand_params()
    assert template.params['option_1'] == 0
    assert not template.params['option_2']

def test_config_get_learnable_params():
    template = MockTemplate()
    params = template.get_learnable_params()
    assert params.get('option_1') == 1
    assert params.get('option_2') is None


def test_equal_templates():
    t1 = MockTemplate()
    t2 = MockTemplate()
    assert t1 == t2

def test_not_equal_templates():
    t1 = MockTemplate(option_1=1)
    t2 = MockTemplate(option_1=2)
    assert t1 != t2
    t1 = MockTemplate()
    t2 = Conv2dTmpl()
    assert t1 != t2

def test_moduletmpl_is_zero_tuple():
    template = MockTemplate(in_shape = (0, 1))
    assert template.is_zero_shape

def test_moduletmpl_is_zero_single():
    template = MockTemplate(in_shape = 1)
    assert not template.is_zero_shape

def test_moduletmpl_is_zero_None():
    template = MockTemplate(in_shape = None)
    assert not template.is_zero_shape

def test_moduletmpl_in_shape_change():
    template = MockTemplate(in_shape=0)
    assert template.out_shape == 0
    template.in_shape = (1, 1)
    assert template.out_shape == (1, 1)
    assert not template.is_zero_shape

def test_moduletmpl_config_change():        
    class MockTemplate(ModuleTemplate_):
        _params = {
            'option': 1
        }
        def _update_out_shape(self):
            self.out_shape = self.params['option']

    template = MockTemplate(in_shape=10)
    assert template.out_shape == 1

    template.params = {'option': 2}
    assert template.out_shape == 2


def test_convtmpl_padding_size():
    template = ConvTemplate_(in_shape=(1, 2, 2), padding=True, kernel_size=(5, 5))
    assert template._calc_padding_size() == (2, 2)

def test_convtmpl_padding_size_zero():
    template = ConvTemplate_(in_shape=(1, 2, 2), padding=False, kernel_size=(5, 5))
    assert template._calc_padding_size() == (0, 0)

def test_convtmpl_out_shape():
    template = ConvTemplate_(in_shape = (1, 4, 4))
    template.params = {
        'out_channels': 32,
        'kernel_size': (3, 3),
        'stride': (1, 1),
        'padding': True,
        'dilation': (1, 1),
        'ceil_mode': False,
    }
    assert template.out_shape == (32, 4, 4)
    template.params['out_channels'] = 16
    template.update_shape()
    assert template.out_shape == (16, 4, 4)
    template.params['padding'] = False
    template.update_shape()
    assert template.out_shape == (16, 2, 2)
    template.params['padding'] = True
    template.params['stride'] = (2, 2)
    template.update_shape()
    assert template.out_shape == (16, 2, 2)
    template.params['stride'] = (5, 5)
    template.update_shape()
    assert template.out_shape == (16, 1, 1)

def test_convtmpl_rand():
    random.seed(0)
    template = ConvTemplate_()
    template.config = {
        'out_channels': {
            'default': 32,
            'range': [2, 4, 8]
        },
        'kernel_size': {
            'default': (3, 3),
            'range': [(1, 1), (2, 2), (3, 3)]
        },
        'stride': {
            'default': (1, 1),
            'range': [(1, 1), (2, 2), (3, 3)]
        },
        'padding': {
            'default': True,
            'range': [True, False]
        },
        'dilation': {
            'default':  (1, 1)
        },
        'ceil_mode': {
            'default': False
        }
    }
    template.gen_rand_params()
    assert template.params['out_channels'] == 4
    assert template.params['kernel_size'] == (2, 2)
    assert template.params['stride'] == (1, 1)
    assert template.params['padding'] == False
    assert template.params['dilation'] == (1, 1)
    assert template.params['ceil_mode'] == False

def test_lintmpl():
    tmpl = LinearTmpl(10, out_features = 32, bias=False)
    assert tmpl.in_shape == 10
    assert tmpl.out_shape == 32
    assert tmpl.params['bias'] is False
    random.seed(0)
    tmpl.gen_rand_params()
    assert tmpl.out_shape == tmpl.params['out_features']
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
    tmpl.gen_rand_params()
    module = tmpl.instantiate_module()
    batch_size = 4
    input = torch.randn((batch_size, 1, 4, 4))
    output = module(input.float())
    assert tmpl.out_shape == (output.shape[1], output.shape[2], output.shape[3])


# def test_seq_tmpl():
#     conv = Conv2dTmpl((3, 4, 4), kernel_size=(3, 3), stride=(2, 2))
#     flat = FlattenTmpl()
#     gap = GlobalAvgPool2dTmpl()
#     lin = LinearTmpl()
#     seq = SequentialEx((3, 4, 4), 10, conv, gap, flat, lin)
#     batch_size = 4
#     input = torch.randn((batch_size, 3, 4, 4))
#     output = seq(input.float())
#     assert (10) == (output.shape[1])


def test_layertmpl_init():
    conv = Conv2dTmpl(out_channels=16, kernel_size=(3, 3), stride=(2, 2))
    flat = FlattenTmpl()
    gap = GlobalAvgPool2dTmpl()
    layer = LayerTemplate_(None, conv, gap, flat)
    assert layer.in_shape == None
    assert layer.out_shape == None
    layer.in_shape = (3, 4, 4)
    assert layer.out_shape == (16)
    layer.templates[0].params['out_channels'] = 32
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
    layer.config = {
        'activation': ['ReLUTmpl'],
        'freq': {
            'MaxPool2dTmpl': 1,
            'BatchNorm2dTmpl': 1,
            'AvgPool2dTmpl': 1,
            'Dropout2dTmpl': 1
        }
    }
    layer.gen_rand_layer()
    assert layer.out_shape is None
    layer.in_shape = (1, 5, 5)
    assert layer.out_shape == (1, 1, 1)


def test_convlayer_tmpl():
    random.seed(0)
    layer = ConvLayerTmpl()
    layer.config = {
        'activation': ['ReLUTmpl', 'LeakyReLUTmpl', 'PReLUTmpl'],
        'freq': {
            'MaxPool2dTmpl': 1,
            'BatchNorm2dTmpl': 1,
            'AvgPool2dTmpl': 1,
            'Dropout2dTmpl': 0
        }
    }
    layer.gen_rand_layer()
    assert layer.out_shape is None
    random.seed(0)
    layer = ConvLayerTmpl((3, 4, 4))
    layer.config = {
        'activation': ['ReLUTmpl', 'LeakyReLUTmpl', 'PReLUTmpl'],
        'freq': {
            'MaxPool2dTmpl': 1,
            'BatchNorm2dTmpl': 1,
            'AvgPool2dTmpl': 1,
            'Dropout2dTmpl': 0,
        }
    }
    layer.gen_rand_layer()
    assert layer.out_shape == (192, 2, 2)
    random.seed(0)
    layer = ConvLayerTmpl((3, 4, 4))
    layer.config = {
        'activation': ['ReLUTmpl', 'LeakyReLUTmpl', 'PReLUTmpl'],
        'freq': {
            'MaxPool2dTmpl': 1,
            'BatchNorm2dTmpl': 1,
            'AvgPool2dTmpl': 1,
            'Dropout2dTmpl': 0
        }
    }
    layer.gen_rand_layer()
    layer_instance = layer.instantiate_layer()
    input = torch.randn((32, 3, 4, 4))
    output = layer_instance(input.float())    
    assert layer.out_shape == (output.shape[1], output.shape[2], output.shape[3])


def test_linlayer_tmpl():
    random.seed(0)
    layer = LinLayerTmpl()
    layer.config = {
        'activation' : ['ReLUTmpl', 'LeakyReLUTmpl', 'PReLUTmpl'],
        'Dropout2dTmpl': 0.5
    }    
    layer.gen_rand_layer()
    assert layer.out_shape is None
    random.seed(0)
    layer = LinLayerTmpl(32)
    layer.config = {
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

def test_model_layers_100():
    for i in range(100):
        test_model_layers_init()