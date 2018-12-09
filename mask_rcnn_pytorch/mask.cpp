#include "mask.h"
#include "roialign.h"

MaskImpl::MaskImpl() {}

MaskImpl::MaskImpl(uint32_t depth,
                   uint32_t pool_size,
                   const std::vector<uint32_t>& image_shape,
                   uint32_t num_classes)
    : padding_(/*kernel_size*/ 3, /*stride*/ 1),
      conv1_(torch::nn::Conv2dOptions(depth,
                                      256,
                                      /*kernel [size, stride]*/ {3, 1})),
      bn1_(torch::nn::BatchNormOptions(256).eps(0.001)),
      conv2_(torch::nn::Conv2dOptions(256,
                                      256,
                                      /*kernel [size, stride]*/ {3, 1})),
      bn2_(torch::nn::BatchNormOptions(256).eps(0.001)),
      conv3_(torch::nn::Conv2dOptions(256,
                                      256,
                                      /*kernel [size, stride]*/ {3, 1})),
      bn3_(torch::nn::BatchNormOptions(256).eps(0.001)),
      conv4_(torch::nn::Conv2dOptions(256,
                                      256,
                                      /*kernel [size, stride]*/ {3, 1})),
      bn4_(torch::nn::BatchNormOptions(256).eps(0.001)),
      conv5_(torch::nn::Conv2dOptions(256,
                                      num_classes,
                                      /*kernel [size, stride]*/ {1, 1})),
      pool_size_(pool_size),
      image_shape_(image_shape) {
  conv_trans_weights_ =
      torch::zeros({256, 256, /*kernel_size*/ 2}, torch::requires_grad());
  conv_trans_weights_ = torch::nn::init::xavier_uniform_(conv_trans_weights_);
  conv_trans_bias_ = torch::zeros({256}, torch::requires_grad());
}

torch::Tensor MaskImpl::forward(at::Tensor x, at::Tensor rois) {
  // x = PyramidRoiAlign(rois + x, pool_size_, image_shape_);
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
  x = torch::conv_transpose2d(x, conv_trans_weights_, conv_trans_bias_,
                              /*stride*/ 2);
  x = torch::relu(x);
  x = conv5_->forward(x);
  x = torch::sigmoid(x);

  return x;
}
