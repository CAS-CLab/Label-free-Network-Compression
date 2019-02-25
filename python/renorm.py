import os
os.environ['GLOG_minloglevel'] = '0'
import numpy as np
import sys
from sys import argv
from utils import *
from config import *

sys.path.append(pycaffe_path)

import caffe
from caffe.proto import caffe_pb2
from google.protobuf.text_format import Parse,MessageToString

## Create BN updated caffemodel
print ">> start renorming ..."

caffe.set_mode_cpu()

net = caffe.Net(BN_update_prototxt, quantized_caffemodel, caffe.TRAIN)
net.forward()

net.save(BN_quantized_caffemodel)

os.remove(quantized_caffemodel)
os.remove(BN_update_prototxt)
