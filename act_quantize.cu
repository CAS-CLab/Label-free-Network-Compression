#include <algorithm>
#include <math.h>
#include <vector>

#include "caffe/layers/act_quantize_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void QuantForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope, Dtype positive_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype tmp = in[index] > Dtype(0) ? Dtype(floor(in[index]/positive_slope+0.5)) : in[index]*negative_slope;
    out[index] = tmp > Dtype(127) ? Dtype(127) : tmp;
  }
}

template <typename Dtype>
void QuantLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype slope = this->layer_param_.relu_param().negative_slope();
  
  // NOLINT_NEXT_LINE(whitespace/operators)
  QuantForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, Dtype(0.), slope);
  CUDA_POST_KERNEL_CHECK;
  
  caffe_gpu_scal(count, slope, top_data);
}

template <typename Dtype>
void QuantLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}


INSTANTIATE_LAYER_GPU_FUNCS(QuantLayer);


}  // namespace caffe
