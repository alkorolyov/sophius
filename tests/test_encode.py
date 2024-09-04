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
        len_model_encoding = len(m.get_templates()[:-1])
        assert vec.shape == (len_model_encoding, ENCODING_SIZE)
        assert len(hash_str) == 8 * len_model_encoding


# def test_encode_decode():
#     model_gen = ConvModelGenerator((3, 32, 32), 10)
#     encoder = Encoder()
#     m = model_gen.generate_model_tmpl()
#     hash_str = encoder.model2hash(m)
#     templates = encoder.decode_template()
#     pass