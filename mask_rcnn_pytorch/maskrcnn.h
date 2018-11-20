#ifndef MASKRCNN_H
#define MASKRCNN_H

#include "config.h"

#include <torch/torch.h>
#include <memory>

class MaskRCNN : public torch::nn::Module {
 public:
  MaskRCNN(std::string model_dir, std::shared_ptr<Config const> config);

  bool detect(const at::Tensor& image);

 private:
  std::string model_dir_;
  std::shared_ptr<Config const> config_;
};

#endif  // MASKRCNN_H
