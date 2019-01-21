#include "mask.h"
#include "debug.h"
#include "roialign.h"

MaskImpl::MaskImpl() {}

MaskImpl::MaskImpl(uint32_t depth,
                   uint32_t pool_size,
                   const std::vector<int32_t>& image_shape,
                   uint32_t num_classes)
    : padding_(/*kernel_size*/ 3, /*stride*/ 1),
      conv1_(torch::nn::Conv2dOptions(depth, 256, 3).stride(1)),
      bn1_(torch::nn::BatchNormOptions(256).eps(0.001)),
      conv2_(torch::nn::Conv2dOptions(256, 256, 3).stride(1)),
      bn2_(torch::nn::BatchNormOptions(256).eps(0.001)),
      conv3_(torch::nn::Conv2dOptions(256, 256, 3).stride(1)),
      bn3_(torch::nn::BatchNormOptions(256).eps(0.001)),
      conv4_(torch::nn::Conv2dOptions(256, 256, 3).stride(1)),
      bn4_(torch::nn::BatchNormOptions(256).eps(0.001)),
      conv5_(torch::nn::Conv2dOptions(256, num_classes, 1).stride(1)),
      deconv_(Deconv()),
      pool_size_(pool_size),
      image_shape_(image_shape) {
  register_module("padding", padding_);
  register_module("conv1", conv1_);
  register_module("bn1", bn1_);
  register_module("conv2", conv2_);
  register_module("bn2", bn2_);
  register_module("conv3", conv3_);
  register_module("bn3", bn3_);
  register_module("conv4", conv4_);
  register_module("bn4", bn4_);
  register_module("conv5", conv5_);
  register_module("deconv", deconv_);
}

torch::Tensor MaskImpl::forward(std::vector<torch::Tensor> feature_maps,
                                at::Tensor rois) {
  feature_maps.insert(feature_maps.begin(), rois);
  auto x = PyramidRoiAlign(feature_maps, pool_size_, image_shape_);
  x = conv1_->forward(padding_->forward(x));
  x = bn1_->forward(x);
  x = torch::relu(x);
  x = conv2_->forward(padding_->forward(x));
  x = bn2_->forward(x);
  x = torch::relu(x);
  x = conv3_->forward(padding_->forward(x));
  x = bn3_->forward(x);
  x = torch::relu(x);
  x = conv4_->forward(padding_->forward(x));
  x = bn4_->forward(x);
  x = torch::relu(x);
  x = deconv_->forward(x);
  x = torch::relu(x);
  x = conv5_->forward(x);
  x = torch::sigmoid(x);

  return x;
}

DeconvImpl::DeconvImpl() {
  conv_trans_weights_ = torch::zeros({256, 256, 2, 2}, torch::requires_grad());
  conv_trans_weights_ = torch::nn::init::xavier_uniform_(conv_trans_weights_);
  conv_trans_bias_ = torch::zeros({256}, torch::requires_grad());

  register_parameter("weight", conv_trans_weights_);
  register_parameter("bias", conv_trans_bias_);
}

at::Tensor DeconvImpl::forward(at::Tensor x) {
  return torch::conv_transpose2d(x, conv_trans_weights_, conv_trans_bias_,
                                 /*stride*/ 2);
}
