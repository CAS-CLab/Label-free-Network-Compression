#include <algorithm>
#include <vector>

#include "caffe/layers/act_quantize_layer.hpp"

namespace caffe {

template <typename Dtype>
void QuantLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i] > Dtype(0.) ? floor(bottom_data[i]/slope + 0.5) : Dtype(0.);
    top_data[i] = top_data[i] > Dtype(127.) ? Dtype(127.) : top_data[i];
  }
}

template <typename Dtype>
void QuantLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}


#ifdef CPU_ONLY
STUB_GPU(QuantLayer);
#endif

INSTANTIATE_CLASS(QuantLayer);
REGISTER_LAYER_CLASS(Quant);
}  // namespace caffe
