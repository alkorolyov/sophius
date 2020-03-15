import torch.nn as nn
from sophius.templates import ConvLayerTmpl, FlatLayerTmpl, LinLayerTmpl, LinearTmpl, GapLayerTmpl, ModelTmpl
import random

################ MODEL GENERATOR ####################

class ModelGenerator_():
    def __init__(self):
        self.layers = []
        self.conv_num = None
        self.lin_num = None
        self.in_shape = None
        self.out_shape = None
    
    def generate_model_tmpl(self):     
        # children override this
        return ModelTmpl()
    
    def _gen_conv_layers(self):
        next_in_shape = self.in_shape
        conv_layers_num = random.randrange(1, self.conv_num + 1)        
        for _ in range(conv_layers_num):
            conv = ConvLayerTmpl(next_in_shape)
            conv.gen_rand_layer()
            self.layers.append(conv)
            next_in_shape = conv.out_shape
            if conv.zero_shape: 
                conv.zero_shape = False
                break

    def _gen_gap_layer(self):
        next_in_shape = self.layers[-1].in_shape    # next shape - previos layer shape
        gap = GapLayerTmpl(next_in_shape)
        gap.gen_rand_layer()
        self.layers.append(gap)

    def _gen_flatten_layer(self):
        next_in_shape = self.layers[-1].in_shape    # next shape - previos layer shape
        flatten = FlatLayerTmpl(next_in_shape)
        flatten.gen_rand_layer()
        self.layers.append(flatten)
    
    def _gen_lin_layers(self):
        next_in_shape = self.layers[-1].in_shape    # next shape - previos layer shape
        lin_layers_num = random.randrange(0, self.lin_num)

        for _ in range(lin_layers_num):
            lin = LinLayerTmpl(next_in_shape)
            lin.gen_rand_layer()
            self.layers.append(lin)
            next_in_shape = lin.out_shape         

    def _gen_lastlin_layer(self):
        next_in_shape = self.layers[-1].in_shape    # next shape - previos layer shape        
        last_tmpl = LinearTmpl(next_in_shape, self.out_shape)
        last_layer = LinLayerTmpl(next_in_shape, last_tmpl)
        self.layers.append(last_layer)

    def generate_model(self):
        m_tmpl = self.generate_model_tmpl()
        model = m_tmpl.instantiate_model()
        return model


class ConvGAPModelGenerator(ModelGenerator_):
    '''
        in_shape:
        [Conv layers] - [Global Average Pooling] - [Flatten] - [Last Linear]
    '''
    def __init__(self, in_shape, out_shape, conv_num = 2):
        self.conv_num = conv_num
        self.in_shape = in_shape
        self.out_shape = out_shape
    
    def generate_model_tmpl(self):
        self.layers = []        

        self._gen_conv_layers()
        self._gen_gap_layer()
        self._gen_flatten_layer()
        self._gen_lastlin_layer()
        
        model_tmpl = ModelTmpl(self.in_shape, self.out_shape, *self.layers)
        return model_tmpl


class ConvFCModelGenerator(ModelGenerator_):

    '''
        [Conv layers] - [Flatten] - [Linear layers - 1] - [Last Linear]
    '''

    def __init__(self, in_shape, out_shape, conv_num = 2, lin_num = 1):        
        self.conv_num = conv_num
        self.lin_num = lin_num
        self.in_shape = in_shape
        self.out_shape = out_shape
    
    def generate_model_tmpl(self):
        self.layers = []        

        self._gen_conv_layers()
        self._gen_flatten_layer()
        self._gen_lin_layers()
        self._gen_lastlin_layer()
        
        model_tmpl = ModelTmpl(self.in_shape, self.out_shape, *self.layers)
        return model_tmpl