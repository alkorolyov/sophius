import sophius
import random
import torch
import numpy as np
from sophius.templates import Conv2dTmpl, FlattenTmpl,GlobalAvgPool2dTmpl,LinearTmpl,SequentialEx, LayerTemplate_
import sophius.utils as utils

conv = Conv2dTmpl(kernel_size=(3, 3), stride=(2, 2))
# utils.print_properties(conv)
flat = FlattenTmpl()
gap = GlobalAvgPool2dTmpl()
lin = LinearTmpl()

layer = LayerTemplate_( (3, 32, 32) , conv, gap, flat, lin)

utils.print_properties(layer)


# print('in_shape', conv.in_shape)
# print('out_shape', conv.out_shape)
# print('config', conv.config)

# conv.config['out_channels'].value = 16
# conv.config['stride'].value = (3, 3)

# print('in_shape', conv.in_shape)
# print('out_shape', conv.out_shape)
# print('config', conv.config)

# conv.gen_rand_config()
# utils.print_nonprivate_properties(conv)

# print('in_shape', conv.in_shape)
# print('out_shape', conv.out_shape)
# print('config', conv.config)

# lin = LinearTmpl((32))
# print('in_shape', lin.in_shape)
# print('out_shape', lin.out_shape)

# utils.print_nonprivate_properties(lin)
# lin.out_shape = 32

# print('in_shape', lin.in_shape)
# print('out_shape', lin.out_shape)
# utils.print_nonprivate_properties(lin)