#ifndef RPN_H
#define RPN_H

#include "nnutils.h"

#include <torch/torch.h>

class RPNImpl : public torch::nn::Module {
 public:
  RPNImpl();
  /*
   * Builds the model of Region Proposal Network.
   *
   * anchors_per_location: number of anchors per pixel in the feature map
   * anchor_stride: Controls the density of anchors. Typically 1 (anchors for
   *                every pixel in the feature map), or 2 (every other pixel).
   * Returns:
   *   rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
   *   rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
   *   rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be applied
   *             to anchors.
   */
  RPNImpl(uint32_t anchors_per_location,
          uint32_t anchor_stride,
          uint32_t depth);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      torch::Tensor x);

 private:
  SamePad2d padding_{nullptr};
  torch::nn::Conv2d conv_shared_{nullptr};
  torch::nn::Functional relu_{nullptr};
  torch::nn::Conv2d conv_class_{nullptr};
  torch::nn::Conv2d conv_bbox_{nullptr};
};

TORCH_MODULE(RPN);

#endif  // RPN_H
