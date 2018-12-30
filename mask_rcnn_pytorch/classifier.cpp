#include "classifier.h"
#include "debug.h"
#include "roialign.h"

ClassifierImpl::ClassifierImpl() {}

ClassifierImpl::ClassifierImpl(uint32_t depth,
                               uint32_t pool_size,
                               const std::vector<int32_t>& image_shape,
                               uint32_t num_classes)
    : conv1_(torch::nn::Conv2dOptions(depth, 1024, pool_size).stride(1)),
      bn1_(torch::nn::BatchNormOptions(1024).eps(0.001).momentum(0.01)),
      conv2_(torch::nn::Conv2dOptions(1024, 1024, 1).stride(1)),
      bn2_(torch::nn::BatchNormOptions(1024).eps(0.001).momentum(0.01)),
      relu_(torch::relu),
      linear_class_(1024, num_classes),
      linear_bbox_(1024, num_classes * 4),
      pool_size_(pool_size),
      image_shape_(image_shape) {
  register_module("conv1", conv1_);
  register_module("bn1", bn1_);
  register_module("conv2", conv2_);
  register_module("bn2", bn2_);
  register_module("relu", relu_);
  register_module("linear_class", linear_class_);
  register_module("linear_bbox", linear_bbox_);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> ClassifierImpl::forward(
    std::vector<at::Tensor> feature_maps,
    at::Tensor rois) {
  feature_maps.insert(feature_maps.begin(), rois);
  auto x = PyramidRoiAlign(feature_maps, pool_size_, image_shape_);
  x = conv1_->forward(x);
  x = bn1_->forward(x);
  x = relu_->forward(x);
  x = conv2_->forward(x);
  x = bn2_->forward(x);
  x = relu_->forward(x);

  x = x.view({-1, 1024});
  auto mrcnn_class_logits = linear_class_->forward(x);
  auto mrcnn_probs = torch::softmax(mrcnn_class_logits, 1);

  auto mrcnn_bbox = linear_bbox_->forward(x);
  mrcnn_bbox = mrcnn_bbox.view({mrcnn_bbox.size(0), -1, 4});

  return {mrcnn_class_logits, mrcnn_probs, mrcnn_bbox};
}
