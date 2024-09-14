import torch
import pytest
import numpy as np
from sophius.modelgen import ConvModelGenerator
from sophius.templates import Conv2dTmpl
from sophius.encode import str_to_vec, vec_to_str, Encoder, TEMPLATES, ENCODING_SIZE


def test_str_vec():
    for i in range(100):
        seq_len = np.random.choice(range(1, 11))
        orig_vec = np.random.randint(0, 2, (seq_len, ENCODING_SIZE))
        hash_str = vec_to_str(orig_vec)
        assert len(hash_str) == seq_len * 8
        vec = str_to_vec(hash_str)
        assert np.equal(vec, orig_vec).all()


def test_encode_tmpl():
    encoder = Encoder()
    for cls in TEMPLATES:
        tmpl = cls()
        vec = encoder.encode_template(tmpl)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (ENCODING_SIZE,)


def test_encode_model():
    model_gen = ConvModelGenerator((3, 32, 32), 10)
    encoder = Encoder()
    for i in range(100):
        m = model_gen.generate_model_tmpl()
        vec = encoder.model2vec(m)
        hash_str = encoder.model2hash(m)
        len_model_encoding = len(m.templates[:-1])
        assert vec.shape == (len_model_encoding, ENCODING_SIZE)
        assert len(hash_str) == 8 * len_model_encoding


def test_encode_decode_tmpl():
    encoder = Encoder()
    t1 = Conv2dTmpl()
    t1.gen_rand_params()
    vec = encoder.encode_template(t1)
    t2 = encoder.decode_template(vec)
    assert t1 == t2


def test_encode_decode_tmpl_100():
    for _ in range(100):
        test_encode_decode_tmpl()

def test_encode_decode_model():
    in_shape = tuple(np.random.randint(2, 10, size = 3))
    out_shape = np.random.randint(2, 10)
    conv_num = np.random.randint(1, 10)
    lin_num = np.random.randint(1, 3)
    batch_size = np.random.randint(1, 10)
    generator = ConvModelGenerator(in_shape, out_shape, conv_num, lin_num)
    orig_model_tmpl = generator.generate_model_tmpl()

    encoder = Encoder()
    hash_str = encoder.model2hash(orig_model_tmpl)
    model_tmpl = encoder.hash2model(hash_str, in_shape=in_shape, out_shape=out_shape)

    assert model_tmpl == orig_model_tmpl

    model = model_tmpl.instantiate_model()
    input = torch.randn(in_shape)[None, :, :, :]
    input = input.expand(batch_size, -1, -1, -1)
    model.eval()
    output = model(input)
    assert model_tmpl.out_shape == out_shape
    assert model_tmpl.out_shape == output.shape[1]

def test_encode_decode_model_100():
    for _ in range(100):
        test_encode_decode_model()



