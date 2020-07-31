#include <vector>

#include "caffe/layers/additive_smoothl1_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void AdditiveSmoothL1Forward(const int n, const Dtype* in, Dtype* out,
    Dtype sigma2) {
  // f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
  //        |x| - 0.5 / sigma / sigma    otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    if (abs_val > 1.0 / sigma2) {
      out[index] = val > 0 ? 1./sigma2 : -1./sigma2;
    }
  }
}

template <typename Dtype>
void AdditiveSmoothL1Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      top[0]->mutable_gpu_data());    // d := b0 - b1
  
  // NOLINT_NEXT_LINE(whitespace/operators)
  AdditiveSmoothL1Forward<Dtype><<<CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS>>>(
      count, top[0]->gpu_data(), top[0]->mutable_gpu_data(), sigma2_);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void AdditiveSmoothL1Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0] || propagate_down[1] ){
    NOT_IMPLEMENTED;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AdditiveSmoothL1Layer);

}  // namespace caffe
