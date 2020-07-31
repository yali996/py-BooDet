#ifndef CAFFE_FOCAL_LOSS_LAYER_HPP_
#define CAFFE_FOCAL_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
class FocalLossLayer : public LossLayer<Dtype> {
 public:
  explicit FocalLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param),
          sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
          sigmoid_output_(new Blob<Dtype>()) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FocalLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }

 protected:
  /// @copydoc SigmoidCrossEntropyLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// Read the normalization mode parameter and compute the normalizer based
  /// on the blob size.  If normalization_mode is VALID, the count of valid
  /// outputs will be read from valid_count, unless it is -1 in which case
  /// all outputs are assumed to be valid.
  virtual Dtype get_normalizer(
      LossParameter_NormalizationMode normalization_mode, int valid_count);


  Blob<Dtype> mul_;  
  Blob<Dtype> prob_;  
  Blob<Dtype> log_prob_;
  float alpha_, beta_, gamma_;
  int count_, num_;
  int num_classes_;

  /// The internal SigmoidLayer used to map predictions to probabilities.
  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  /// sigmoid_output stores the output of the SigmoidLayer.
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  /// bottom vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  /// top vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_top_vec_;
    /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  /// How to normalize the output loss.
  LossParameter_NormalizationMode normalization_;
};

}  // namespace caffe

#endif  // CAFFE_FOCAL_LOSS_LAYER_HPP_
