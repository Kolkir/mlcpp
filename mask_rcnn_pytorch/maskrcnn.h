#ifndef MASKRCNN_H
#define MASKRCNN_H

#include "classifier.h"
#include "config.h"
#include "fpn.h"
#include "mask.h"
#include "rpn.h"

#include <torch/torch.h>
#include <memory>

class MaskRCNNImpl : public torch::nn::Module {
 public:
  MaskRCNNImpl(std::string model_dir, std::shared_ptr<Config const> config);

  bool Detect(const std::vector<at::Tensor>& images);

 private:
  void Build();
  void InitializeWeights();
  std::tuple<at::Tensor, at::Tensor, at::Tensor> MoldInputs(
      const std::vector<at::Tensor>& images);

 private:
  std::string model_dir_;
  std::shared_ptr<Config const> config_;

  FPN fpn_{nullptr};
  torch::Tensor anchors_;
  RPN rpn_{nullptr};
  Classifier classifier_{nullptr};
  Mask mask_{nullptr};
};

TORCH_MODULE(MaskRCNN);

#endif  // MASKRCNN_H
