import sophius
import random
from sophius.templates import LinearTmpl

def test_lintmpl():
    lin = LinearTmpl(10, 128, bias = False)
    assert lin.in_shape == 10
    assert lin.out_shape == 128
    assert lin.params['bias'].value == False    
    random.seed(0)
    lin.gen_params()
    lin.calc_out_shape()
    assert lin.out_shape == lin.params['out_features'].value
    assert lin.out_shape == 2048    
    lin.instantiate_module()
    

