pycaffe_path = "/home/caffe-master/python" # /your/caffe/python/path

model_name = "resnet-18"

train_dataset = "/data/ilsvrc12_lmdb_shrt_256/ilsvrc12_train_lmdb"
val_dataset = "/data/ilsvrc12_lmdb_shrt_256/ilsvrc12_val_lmdb"

BN_update_batch_size = 1000 # using 1K images to update BN
activation_quantization_batch_size = 400 # using 0.4K images to quantize activations 
val_batch_size = 50 # test batch size
val_data_size = 50000 # 5W images in validation set 

if model_name=="alexnet":
    full_precision_caffemodel = '../alexnet_bn/alexnet_bn.caffemodel' # /your/full/precision/caffe/model/path
    full_precision_prototxt = '../qmodel/alexnet_bn.prototxt' # /your/full/precision/caffe/prototxt/path
    BN_update_prototxt = '../qmodel/alexnet_bn_BN_update.prototxt' # tmp file
    act_int8_prototxt = '../qmodel/alexnet_bn_act_int8.prototxt' # tmp file
    quantized_caffemodel = '../qmodel/alexnet_bn_quantized.caffemodel' # tmp file
    BN_quantized_caffemodel = '../qmodel/refined_alexnet_bn_quantized.caffemodel' # final model
    val_prototxt = '../qmodel/refined_alexnet_bn_quantized.prototxt' # final prototxt
elif model_name=="vgg16_bn":
    full_precision_caffemodel = '../vgg16_bn/vgg16_bn.caffemodel'
    full_precision_prototxt = '../qmodel/vgg16_bn.prototxt'
    BN_update_prototxt = '../qmodel/vgg16_bn_BN_update.prototxt'
    act_int8_prototxt = '../qmodel/vgg16_bn_act_int8.prototxt'
    quantized_caffemodel = '../qmodel/vgg16_bn_quantized.caffemodel'
    BN_quantized_caffemodel = '../qmodel/refined_vgg16_bn_quantized.caffemodel'
    val_prototxt = '../qmodel/refined_vgg16_bn_quantized.prototxt'
elif model_name=="resnet-18":
    full_precision_caffemodel = '../resnet-18/resnet-18.caffemodel'
    full_precision_prototxt = '../qmodel/resnet-18.prototxt'
    BN_update_prototxt = '../qmodel/resnet-18_BN_update.prototxt'
    act_int8_prototxt = '../qmodel/resnet-18_act_int8.prototxt'
    quantized_caffemodel = '../qmodel/resnet-18_quantized.caffemodel'
    BN_quantized_caffemodel = '../qmodel/refined_resnet-18_quantized.caffemodel'
    val_prototxt = '../qmodel/refined_resnet-18_quantized.prototxt'
elif model_name=="resnet-50":
    full_precision_caffemodel = '../resnet-50/resnet-50.caffemodel'
    full_precision_prototxt = '../qmodel/resnet-50.prototxt'
    BN_update_prototxt = '../qmodel/resnet-50_BN_update.prototxt'
    act_int8_prototxt = '../qmodel/resnet-50_act_int8.prototxt'
    quantized_caffemodel = '../qmodel/resnet-50_quantized.caffemodel'
    BN_quantized_caffemodel = '../qmodel/refined_resnet-50_quantized.caffemodel'
    val_prototxt = '../qmodel/refined_resnet-50_quantized.prototxt'
else:
    assert True==False, "Unknown model name"

print ">> MODEL : " + model_name
print ">> Final 4-8-bit MODEL : " + BN_quantized_caffemodel
print ">> Final 4-8-bit Prototxt : " + val_prototxt
