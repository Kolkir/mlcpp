#ifndef NNUTILS_H
#define NNUTILS_H

#include <torch/torch.h>
#include <vector>

class SamePad2d : public torch::nn::Module {
 public:
  SamePad2d(uint32_t kernel_size, uint32_t stride);

  torch::Tensor forward(torch::Tensor input);

 private:
  uint32_t kernel_size_;
  uint32_t stride_;
};

#endif  // NNUTILS_H
