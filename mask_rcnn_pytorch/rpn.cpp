#include "rpn.h"
#include "debug.h"

RPNImpl::RPNImpl() {}

RPNImpl::RPNImpl(uint32_t anchors_per_location,
                 uint32_t anchor_stride,
                 uint32_t depth)
    : padding_(/*kernel_size*/ 3, /*stride*/ anchor_stride),
      conv_shared_(
          torch::nn::Conv2dOptions(depth, 512, 3).stride(anchor_stride)),
      relu_(torch::relu),
      conv_class_(
          torch::nn::Conv2dOptions(512, 2 * anchors_per_location, 1).stride(1)),

      conv_bbox_(torch::nn::Conv2dOptions(512, 4 * anchors_per_location, 1)
                     .stride(1)) {
  register_module("padding", padding_);
  register_module("conv_shared", conv_shared_);
  register_module("relu", relu_);
  register_module("conv_class", conv_class_);
  register_module("conv_bbox", conv_bbox_);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> RPNImpl::forward(at::Tensor x) {
  // Shared convolutional base of the RPN
  x = relu_->forward(conv_shared_->forward(padding_->forward(x)));

  // Anchor Score. [batch, anchors per location * 2, height, width].
  auto rpn_class_logits = conv_class_->forward(x);

  // Reshape to [batch, 2, anchors]
  rpn_class_logits = rpn_class_logits.permute({0, 2, 3, 1});
  rpn_class_logits = rpn_class_logits.contiguous();
  rpn_class_logits = rpn_class_logits.view({x.size(0), -1, 2});

  // Softmax on last dimension of BG/FG.
  auto rpn_probs = torch::softmax(rpn_class_logits, /*dim*/ 2);

  // Bounding box refinement. [batch, H, W, anchors per location, depth]
  // where depth is [x, y, log(w), log(h)]
  auto rpn_bbox = conv_bbox_->forward(x);

  // Reshape to [batch, 4, anchors]
  rpn_bbox = rpn_bbox.permute({0, 2, 3, 1});
  rpn_bbox = rpn_bbox.contiguous();
  rpn_bbox = rpn_bbox.view({x.size(0), -1, 4});

  return {rpn_class_logits, rpn_probs, rpn_bbox};
}
