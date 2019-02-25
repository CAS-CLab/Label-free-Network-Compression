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

print ">> start activation quantization ..."

caffe.set_mode_cpu()

net_param = caffe_pb2.NetParameter()
with open(act_int8_prototxt, 'rt') as f:
    Parse(f.read(), net_param)
layer_num = len(net_param.layer)

relu_top = []
relu_layer_idx = []

for layer_idx in range(layer_num):
    layer = net_param.layer[layer_idx]
    if layer.type=='ReLU':
        relu_top.append(layer.top[0])
        relu_layer_idx.append(layer_idx)

del net_param

net = caffe.Net(act_int8_prototxt, BN_quantized_caffemodel, caffe.TRAIN)
net.forward()

alpha_list = []

for top in relu_top:
    act_float32 = net.blobs[top].data[...]
    act_float32 = act_float32.astype(numpy.float32)
    act_float32 = act_float32.ravel()

    print ">> processing " + top

    epsilon = 1e-3
    prev_alpha = 1e10
    alpha = max(act_float32)/127
    act_uint8 = np.floor(act_float32/alpha+0.5)
    np.clip(act_uint8, 0, 127, out=act_uint8)
    diff = abs(prev_alpha-alpha)
    div = diff/alpha
    prev_alpha = alpha
    pre_loss = 1e10
    i = 1
    while div > numpy.float64(epsilon):
        act_uint8 = act_uint8.astype(numpy.float32)
        alpha = numpy.dot(act_float32,act_uint8) / numpy.dot(act_uint8,act_uint8)
        act_uint8 = np.floor(act_float32/alpha+0.5)
        np.clip(act_uint8, 0, 127, out=act_uint8)
        diff = abs(prev_alpha-alpha)
        div = diff/alpha
        loss = numpy.linalg.norm(act_float32-alpha*act_uint8,2)
        print "iter ", i, " | alpha", alpha, " | loss ", loss
        i += 1
        if loss > pre_loss:
            break
        pre_loss = loss
        prev_alpha = alpha
    alpha_list.append(prev_alpha)

## Create val prototxt
new_net_param = caffe_pb2.NetParameter()
with open(act_int8_prototxt, 'rt') as f:
    Parse(f.read(), new_net_param)

for i in range(len(relu_layer_idx)):
    layer = new_net_param.layer[relu_layer_idx[i]]
    layer.relu_param.negative_slope = alpha_list[i]
    layer.type = 'Quant'

layer_num = len(new_net_param.layer)
for layer_idx in range(layer_num):
    layer = new_net_param.layer[layer_idx]
    if layer.type=='Data':
        layer.data_param.batch_size = val_batch_size
        layer.data_param.source = val_dataset

with open(val_prototxt, 'wt') as f:
    f.write(MessageToString(new_net_param))

os.remove(act_int8_prototxt)

# print "final prototxt : " + val_prototxt