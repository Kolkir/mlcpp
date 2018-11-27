#ifndef RESNET_H
#define RESNET_H

#include "nnutils.h"

#include <torch/torch.h>
#include <vector>

class Bottleneck : public torch::nn::Module {
 public:
  Bottleneck(uint32_t inplanes,
             uint32_t planes,
             uint32_t stride = 1,
             torch::nn::Sequential downsample = nullptr);

  static constexpr uint32_t expansion{4};

  torch::Tensor forward(torch::Tensor x);

 private:
  torch::nn::Conv2d conv1_;
  torch::nn::BatchNorm bn1_;
  SamePad2d padding2_;
  torch::nn::Conv2d conv2_;
  torch::nn::BatchNorm bn2_;
  torch::nn::Conv2d conv3_;
  torch::nn::BatchNorm bn3_;
  torch::nn::Functional relu_;
  torch::nn::Sequential downsample_;
};

class ResNet : public torch::nn::Module {
 public:
  enum Architecture { ResNet50, ResNet101 };

  ResNet(Architecture architecture, bool stage5);

  torch::Tensor forward(torch::Tensor input);

  auto GetStages() { return std::make_tuple(c1_, c2_, c3_, c4_, c5_); }

 private:
  torch::nn::Sequential MakeLayer(uint32_t planes,
                                  uint32_t blocks,
                                  uint32_t stride = 1);

 private:
  uint32_t inplanes_{64};
  std::vector<uint32_t> layers_{3, 4, 0, 3};
  bool stage5_{false};

  torch::nn::Sequential c1_;
  torch::nn::Sequential c2_;
  torch::nn::Sequential c3_;
  torch::nn::Sequential c4_;
  torch::nn::Sequential c5_;
};

#endif  // RESNET_H
