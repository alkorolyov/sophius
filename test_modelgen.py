import random
import pytest

import numpy as np
import torch
import torch.nn as nn

import sophius
from sophius.modelgen import ConvGAPModelGenerator, ConvFCModelGenerator


def test_model_convgap():
    in_shape = tuple(np.random.randint(2, 10, size = 3))
    out_shape = np.random.randint(1, 10)
    conv_num = np.random.randint(1, 10)
    batch_size = np.random.randint(1, 10)
    generator = ConvGAPModelGenerator(in_shape, out_shape, conv_num)
    model_tmpl = generator.generate_model_tmpl()
    model = model_tmpl.instantiate_model()
    input = torch.randn(in_shape)[None, :, :, :]
    input = input.expand(batch_size, -1, -1, -1)
    model.eval()
    output = model(input)
    assert model_tmpl.out_shape == out_shape
    assert model_tmpl.out_shape == output.shape[1]


def test_100_models_convgap():
    for _ in range(100):
        test_model_convgap()


def test_model_convfc():
    in_shape = tuple(np.random.randint(2, 10, size = 3))
    out_shape = np.random.randint(2, 10)
    conv_num = np.random.randint(1, 10)
    lin_num = np.random.randint(1, 3)
    batch_size = np.random.randint(1, 10)
    generator = ConvFCModelGenerator(in_shape, out_shape, conv_num, lin_num)
    model_tmpl = generator.generate_model_tmpl()
    model = model_tmpl.instantiate_model()
    input = torch.randn(in_shape)[None, :, :, :]
    input = input.expand(batch_size, -1, -1, -1)
    model.eval()
    output = model(input)
    assert model_tmpl.out_shape == out_shape
    assert model_tmpl.out_shape == output.shape[1]

def test_100_models_convfc():
    for _ in range(100):
        test_model_convfc()
