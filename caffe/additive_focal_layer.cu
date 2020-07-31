#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/additive_focal_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void ReductionAdditiveFocalForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* label, Dtype* target, Dtype* weight,
          const bool has_ignore_label_, const int ignore_label_, const float alpha, const float zmax_, Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int label_value = static_cast<int>(label[index]);
    // if (has_ignore_label_ && label_value == ignore_label_) {
    if (label_value < 0) {
      target[index] = 0;
      weight[index] = 0;
      counts[index] = 0;
    } else {
      // weight[index] *= ( (1.-alpha) * (1.-label_value ) + alpha * label_value );
      Dtype target_value = target[index];
      if (target_value > zmax_) 
        target[index] = zmax_;
      else if(target_value < -zmax_)    
        target[index] = -zmax_;
      counts[index] = label_value > 0 ? 1 : 0;
    }
  }
}


template <typename Dtype>
void AdditiveFocalLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 
  const Dtype* bottom_labels = bottom[1]->gpu_data();
  const int counts = bottom[0]->count();
  Dtype* top_targets = top[0]->mutable_gpu_data();
  Dtype* top_weights = top[1]->mutable_gpu_data();
  Dtype * input_data = prob_.mutable_gpu_data();  // gamma * imput + beta
  Dtype * count_data = prob_.mutable_gpu_diff();  // gamma * imput + beta

  caffe_gpu_set(bottom[0]->count(), Dtype(-1.0), top_targets);
  caffe_gpu_axpy(bottom[0]->count(), Dtype(2.0), bottom[1]->gpu_data(), top_targets); // (0,1)->(-1,1)  
  caffe_gpu_mul(bottom[0]->count(), bottom[0]->gpu_data(), top_targets, input_data); // y * input
  caffe_gpu_scal(bottom[0]->count(), Dtype(gamma_), input_data);    // gamma * y * input
  caffe_gpu_add_scalar(bottom[0]->count(), Dtype(beta_), input_data);  // gamma * y * input + beta
  sigmoid_bottom_vec_[0] = &prob_;
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);

  const Dtype* prob_data = sigmoid_output_->gpu_data();
  caffe_gpu_set(counts, Dtype(1.0), top_weights);
  caffe_gpu_axpy(counts, Dtype(-1.0), prob_data, top_weights);  // 1 - p(x)
  caffe_gpu_mul(counts, prob_data, top_weights, top_weights);   // p(x) * (1-p(x))
 
  caffe_gpu_mul(counts, top_targets, prob_data, top_targets); // (y* - p(x))/(p(x)*(1-p(x)))
  caffe_gpu_powx(counts, top_targets, Dtype(-1.0), top_targets); // (y* - p(x))/(p(x)*(1-p(x)))
  caffe_gpu_scal(counts, Dtype(1.0/gamma_), top_targets);    // gamma * y * input
  
  ReductionAdditiveFocalForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(counts),
      CAFFE_CUDA_NUM_THREADS>>>(counts, input_data, bottom_labels, top_targets, top_weights,
      has_ignore_label_, ignore_label_, alpha_, zmax_, count_data);

  Dtype valid_count = -1;
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    caffe_gpu_asum(counts, count_data, &valid_count);
    // caffe_gpu_asum(counts, bottom_labels, &valid_count);
  }
  Dtype norm_value = get_normalizer(normalization_, valid_count);

  caffe_gpu_scal(counts, Dtype(1.0/norm_value), top_weights);    // gamma * y * input

}



template <typename Dtype>
void AdditiveFocalLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
   
  if (propagate_down[0] || propagate_down[1] ){
    NOT_IMPLEMENTED;
  }

}


INSTANTIATE_LAYER_GPU_FUNCS(AdditiveFocalLayer);


}  // namespace caffe
