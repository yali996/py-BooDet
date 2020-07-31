#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/additive_focal_layer.hpp"

namespace caffe {

template <typename Dtype>
void AdditiveFocalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  gamma_ = this->layer_param_.focal_loss_param().gamma();
  alpha_ = this->layer_param_.focal_loss_param().alpha();
  beta_ = this->layer_param_.focal_loss_param().beta();

  zmax_ = 2.0;	
 
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(&prob_);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }

  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }

  iterations_ = 0;
  top[0]->ReshapeLike(*bottom[0]);
  top[1]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void AdditiveFocalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  prob_.ReshapeLike(*bottom[0]);
  count_ = bottom[0]->count();
  num_ = bottom[0]->num();

  top[0]->ReshapeLike(*bottom[0]);
  top[1]->ReshapeLike(*bottom[0]);
}


template <typename Dtype>
Dtype AdditiveFocalLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(count_);
      break;
    case LossParameter_NormalizationMode_VALID:
        normalizer = Dtype(std::max(valid_count,100)); 
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
       normalizer = Dtype(num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  return std::max(Dtype(1.0), normalizer);
}


template <typename Dtype>
void AdditiveFocalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_labels = bottom[1]->cpu_data();
  const int counts = bottom[0]->count();  
  Dtype* top_targets = top[0]->mutable_cpu_data();
  Dtype* top_weights = top[1]->mutable_cpu_data();
  Dtype * input_data = prob_.mutable_cpu_data();  // gamma * imput + beta
  
  caffe_set(bottom[0]->count(), Dtype(-1.0), top_targets);
  caffe_axpy(bottom[0]->count(), Dtype(2.0), bottom[1]->cpu_data(), top_targets); // (0,1)->(-1,1)  
  caffe_mul(bottom[0]->count(), bottom[0]->cpu_data(), top_targets, input_data); // y * input
  caffe_scal(bottom[0]->count(), Dtype(gamma_), input_data);    // gamma * y * input
  caffe_add_scalar(bottom[0]->count(), Dtype(beta_), input_data);  // gamma * y * input + beta
  sigmoid_bottom_vec_[0] = &prob_;

  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  const Dtype* prob_data = sigmoid_output_->cpu_data();
 
  caffe_set(counts, Dtype(1.0), top_weights);
  caffe_axpy(counts, Dtype(-1.0), prob_data, top_weights);  // 1 - p(x)
  caffe_mul(counts, prob_data, top_weights, top_weights);   // p(x) * (1-p(x))
 
  // caffe_copy(counts, prob_data, top_targets); 
  // caffe_add_scalar(counts, Dtype(0.000001), top_targets); 
  caffe_mul(counts, top_targets, prob_data, top_targets); // (y* - p(x))/(p(x)*(1-p(x)))
  caffe_powx(counts, top_targets, Dtype(-1.0), top_targets); // (y* - p(x))/(p(x)*(1-p(x)))
  caffe_scal(counts, Dtype(1.0/gamma_), top_targets);    // gamma * y * input
 
  int valid_count = caffe_cpu_asum(counts, bottom_labels);
  Dtype norm_value = get_normalizer(normalization_, valid_count);   
  caffe_scal(counts, Dtype(1.0/norm_value), top_weights);    // gamma * y * input
  
 
  for( int i = 0; i < counts; i++ ){
    if(top_targets[i] > zmax_ )
      top_targets[i] = zmax_;
    else if(top_targets[i] < -zmax_ )
      top_targets[i] = -zmax_;
    
    if(bottom_labels[i] < 0 )
      top_weights[i] = 0;
  }
}

template <typename Dtype>
void AdditiveFocalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0] || propagate_down[1] ){
    NOT_IMPLEMENTED;
  }
}

#ifdef CPU_ONLY
 STUB_GPU(AdditiveFocalLayer);
 // STUB_GPU_FORWARD(AdditiveFocalLayer, Forward_gpu);
#endif

INSTANTIATE_CLASS(AdditiveFocalLayer);
REGISTER_LAYER_CLASS(AdditiveFocal);

}  // namespace caffe
