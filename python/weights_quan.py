import os
os.environ['GLOG_minloglevel'] = '2'
import numpy as np
import sys
from sys import argv
from utils import *
from config import *

sys.path.append(pycaffe_path)

import caffe
from caffe.proto import caffe_pb2
from google.protobuf.text_format import Parse,MessageToString

## Create BN update prototxt
net_param = caffe_pb2.NetParameter()
with open(full_precision_prototxt, 'rt') as f:
    Parse(f.read(), net_param)
layer_num = len(net_param.layer)

parameter_layer_idx = []

for layer_idx in range(layer_num):
    layer = net_param.layer[layer_idx]
    if layer.type=='Convolution' or layer.type=='InnerProduct' or layer.type=='DepthwiseConvolution':
        parameter_layer_idx.append(layer_idx)

int8_layers = [parameter_layer_idx[0], parameter_layer_idx[-1]]

new_net_param = caffe_pb2.NetParameter()
new_net_param.name = net_param.name

for layer_idx in range(layer_num):
    layer = net_param.layer[layer_idx]
    new_net_param.layer.extend([layer])
    if layer.type=='Data':
        new_net_param.layer[-1].data_param.batch_size = BN_update_batch_size
        new_net_param.layer[-1].data_param.source = train_dataset
    if layer.type=='BatchNorm':
        new_net_param.layer[-1].batch_norm_param.moving_average_fraction = 0
        new_net_param.layer[-1].batch_norm_param.use_global_stats = False
    if layer.type=='Convolution':
        new_net_param.layer.add()
        new_net_param.layer[-1].name = net_param.layer[layer_idx].name + '_alpha_term'
        new_net_param.layer[-1].type = 'Scale'
        new_net_param.layer[-1].bottom.append(net_param.layer[layer_idx].top[0])
        new_net_param.layer[-1].top.append(net_param.layer[layer_idx].top[0])
        try:
            if new_net_param.layer[-2].convolution_param.bias_term == False:
                new_net_param.layer[-1].scale_param.bias_term = False
            else:
                new_net_param.layer[-1].scale_param.bias_term = True
        except:
            new_net_param.layer[-1].scale_param.bias_term = True
    if layer.type=='InnerProduct':
        new_net_param.layer.add()
        new_net_param.layer[-1].name = net_param.layer[layer_idx].name + '_alpha_term'
        new_net_param.layer[-1].type = 'Scale'
        new_net_param.layer[-1].bottom.append(net_param.layer[layer_idx].top[0])
        new_net_param.layer[-1].top.append(net_param.layer[layer_idx].top[0])
        try:
            if new_net_param.layer[-2].inner_product_param.bias_term == False:
                new_net_param.layer[-1].scale_param.bias_term = False
            else:
                new_net_param.layer[-1].scale_param.bias_term = True
        except:
            new_net_param.layer[-1].scale_param.bias_term = True

with open(BN_update_prototxt, 'wt') as f:
    f.write(MessageToString(new_net_param))

del new_net_param
del net_param

# print "update BN prototxt : " + BN_update_prototxt

## Create activation quantization prototxt

act_net_param = caffe_pb2.NetParameter()
with open(BN_update_prototxt, 'rt') as f:
    Parse(f.read(), act_net_param)
layer_num = len(act_net_param.layer)

for layer_idx in range(layer_num):
    layer = act_net_param.layer[layer_idx]
    if layer.type=='Data':
        layer.data_param.batch_size = activation_quantization_batch_size
        layer.data_param.source = train_dataset
    if layer.type=='BatchNorm':
        layer.batch_norm_param.moving_average_fraction = 0.999
        layer.batch_norm_param.use_global_stats = True

with open(act_int8_prototxt, 'wt') as f:
    f.write(MessageToString(act_net_param))

del act_net_param

# print "activation quantization prototxt : " + act_int8_prototxt


## Create quantized caffemodel
caffe.set_mode_cpu()
# caffe.set_device(7)

net_param = caffe_pb2.NetParameter()
with open(full_precision_prototxt, 'rt') as f:
    Parse(f.read(), net_param)
layer_num = len(net_param.layer)

new_net = caffe.Net(BN_update_prototxt, caffe.TEST)
net = caffe.Net(full_precision_prototxt, full_precision_caffemodel, caffe.TEST)
# Nonparam Layer
excluded_layers = ['Input', 'Data', 'ReLU', 'Pooling', 'Dropout', 'Accuracy', 'Softmax', \
                    'SoftmaxWithLoss', 'Eltwise', 'Concat', 'PReLU', 'LRN']

for layer_idx in range(layer_num):
    layer = net_param.layer[layer_idx]
    if layer.type in excluded_layers:
        continue
    quantized_params = net.params[layer.name]
    new_params = new_net.params[layer.name]
    if layer.type=='Convolution' or layer.type=='InnerProduct' or layer.type=='DepthwiseConvolution':
        print ">> Conv/FC " + layer.name
        if len(quantized_params) == 1:
            new_next_params = new_net.params[layer.name+"_alpha_term"]
            if layer_idx in int8_layers:
                Q,alpha = quantization_with_scale(quantized_params[0].data, new_next_params[0].data, qtype="fixed")
            else:
                Q,alpha = quantization_with_scale(quantized_params[0].data, new_next_params[0].data, qtype="power")
            new_params[0].data[...] = Q
            new_next_params[0].data[...] = alpha
        else:
            new_next_params = new_net.params[layer.name+"_alpha_term"]
            if layer_idx in int8_layers:
                Q,alpha = quantization_with_scale(quantized_params[0].data, new_next_params[0].data, qtype="fixed")
            else:
                Q,alpha = quantization_with_scale(quantized_params[0].data, new_next_params[0].data, qtype="power")
            new_params[0].data[...] = Q
            new_next_params[0].data[...] = alpha
            new_params[1].data[...] = quantized_params[1].data[...]
            new_next_params[1].data[...] = quantized_params[1].data[...]*(1-new_next_params[0].data[...])
    else:
        print ">> BN/Scale " + layer.name
        for i in range(0,len(quantized_params)):
            new_params[i].data[...] = quantized_params[i].data[...]

new_net.save(quantized_caffemodel)

# print "quantized caffemodel : " + quantized_caffemodel

del new_next_params
del net_param
del new_net
del net

