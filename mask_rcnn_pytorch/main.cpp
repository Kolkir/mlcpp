#include <torch/torch.h>
#include <iostream>

int main(int argc [[maybe_unused]], char** argv [[maybe_unused]]) {
  at::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
  return 0;
}
