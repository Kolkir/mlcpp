#include "classifier.h"
#include "roialign.h"

ClassifierImpl::ClassifierImpl() {}

ClassifierImpl::ClassifierImpl(uint32_t depth,
                               uint32_t pool_size,
                               const std::vector<uint32_t>& image_shape,
                               uint32_t num_classes)
    :

      conv1_(
          torch::nn::Conv2dOptions(depth,
                                   1024,
                                   /*kernel [size, stride]*/ {pool_size, 1})),
      bn1_(torch::nn::BatchNormOptions(1024).eps(0.001).momentum(0.01)),
      conv2_(torch::nn::Conv2dOptions(1024,
                                      1024,
                                      /*kernel [size, stride]*/ {1, 1})),
      bn2_(torch::nn::BatchNormOptions(1024).eps(0.001).momentum(0.01)),
      relu_(torch::relu),
      linear_class_(1024, num_classes),
      linear_bbox_(1024, num_classes * 4),
      pool_size_(pool_size),
      image_shape_(image_shape) {}

std::tuple<at::Tensor, at::Tensor, at::Tensor> ClassifierImpl::forward(
    at::Tensor x,
    at::Tensor rois) {
  // x = PyramidRoiAlign(rois + x, pool_size_, image_shape_);
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
