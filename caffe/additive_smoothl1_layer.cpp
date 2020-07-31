#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/additive_smoothl1_layer.hpp"

namespace caffe {

template <typename Dtype>
void AdditiveSmoothL1Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 
  SmoothL1LossParameter loss_param = this->layer_param_.smooth_l1_loss_param();
  sigma2_ = loss_param.sigma() * loss_param.sigma();
  
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void AdditiveSmoothL1Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void AdditiveSmoothL1Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_targets = bottom[1]->cpu_data();
  Dtype* top_targets = top[0]->mutable_cpu_data();
  const int counts = bottom[0]->count();  
  
  caffe_set(counts, Dtype(.0), top_targets);
  caffe_sub(counts, bottom_targets, bottom_data, top_targets);
 
  for( int i = 0; i < counts; i++ ){
    if( abs(top_targets[i]) > 1.0 / sigma2_ )
      top_targets[i] = top_targets[i] > 0 ? 1./sigma2_ : -1./sigma2_;
  }
}

template <typename Dtype>
void AdditiveSmoothL1Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0] || propagate_down[1] ){
    NOT_IMPLEMENTED;
  }
}

#ifdef CPU_ONLY
 STUB_GPU(AdditiveSmoothL1Layer);
 // STUB_GPU(AdditiveSmoothL1Layer, Forward_gpu);
#endif

INSTANTIATE_CLASS(AdditiveSmoothL1Layer);
REGISTER_LAYER_CLASS(AdditiveSmoothL1);

}  // namespace caffe
