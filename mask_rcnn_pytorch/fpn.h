#ifndef FPN_H
#define FPN_H

#include <torch/torch.h>

class FPNImpl : public torch::nn::Module {
 public:
  FPNImpl();
  FPNImpl(torch::nn::Sequential c1,
          torch::nn::Sequential c2,
          torch::nn::Sequential c3,
          torch::nn::Sequential c4,
          torch::nn::Sequential c5,
          uint32_t out_channels);

  std::tuple<torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor>
  forward(torch::Tensor x);

 private:
  torch::nn::Sequential c1_{nullptr};
  torch::nn::Sequential c2_{nullptr};
  torch::nn::Sequential c3_{nullptr};
  torch::nn::Sequential c4_{nullptr};
  torch::nn::Sequential c5_{nullptr};

  torch::nn::Functional p6_{nullptr};
  torch::nn::Conv2d p5_conv1_{nullptr};
  torch::nn::Sequential p5_conv2_{nullptr};
  torch::nn::Conv2d p4_conv1_{nullptr};
  torch::nn::Sequential p4_conv2_{nullptr};
  torch::nn::Conv2d p3_conv1_{nullptr};
  torch::nn::Sequential p3_conv2_{nullptr};
  torch::nn::Conv2d p2_conv1_{nullptr};
  torch::nn::Sequential p2_conv2_{nullptr};
};

TORCH_MODULE(FPN);

#endif  // FPN_H
