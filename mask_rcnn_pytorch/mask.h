#ifndef MASK_H
#define MASK_H

#include "nnutils.h"

#include <inttypes.h>

#include <torch/torch.h>

class MaskImpl : public torch::nn::Module {
 public:
  MaskImpl();
  MaskImpl(uint32_t depth,
           uint32_t pool_size,
           const std::vector<int32_t>& image_shape,
           uint32_t num_classes);

  torch::Tensor forward(torch::Tensor x, torch::Tensor rois);

 private:
  SamePad2d padding_{nullptr};
  torch::nn::Conv2d conv1_{nullptr};
  torch::nn::BatchNorm bn1_{nullptr};
  torch::nn::Conv2d conv2_{nullptr};
  torch::nn::BatchNorm bn2_{nullptr};
  torch::nn::Conv2d conv3_{nullptr};
  torch::nn::BatchNorm bn3_{nullptr};
  torch::nn::Conv2d conv4_{nullptr};
  torch::nn::BatchNorm bn4_{nullptr};
  torch::nn::Conv2d conv5_{nullptr};

  uint32_t pool_size_{0};
  std::vector<int32_t> image_shape_;
  torch::Tensor conv_trans_weights_;
  torch::Tensor conv_trans_bias_;
};

TORCH_MODULE(Mask);

#endif  // MASK_H
