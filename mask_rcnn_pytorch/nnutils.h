#ifndef NNUTILS_H
#define NNUTILS_H

#include <torch/torch.h>
#include <vector>

at::Tensor upsample(at::Tensor x, float scale_factor);
at::Tensor unique1d(at::Tensor tensor);
at::Tensor intersect1d(at::Tensor tensor1, at::Tensor tensor2);

class SamePad2dImpl : public torch::nn::Module {
 public:
  SamePad2dImpl();
  SamePad2dImpl(uint32_t kernel_size, uint32_t stride);

  torch::Tensor forward(torch::Tensor input);

 private:
  uint32_t kernel_size_{0};
  uint32_t stride_{0};
};

TORCH_MODULE(SamePad2d);

#endif  // NNUTILS_H
