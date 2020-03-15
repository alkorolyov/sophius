import sophius
import random
import torch
from sophius.templates import LinearTmpl
import sophius.utils as utils

random.seed(0)
torch.manual_seed(0)
lin = LinearTmpl(10, 128, bias = False)
# utils.print_nonprivate_properties(lin)
lin = LinearTmpl()

lin.gen_params()
lin.calc_out_shape()
utils.print_nonprivate_properties(lin)
lin_pytorch = lin.instantiate_module()

# print(lin)
