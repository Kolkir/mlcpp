#include <torch/torch.h>
#include <iostream>

// Define a new Module.
class Net : torch::nn::Module {
 public:
  Net() {
    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(8, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 1));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, /*p=*/0.5, is_training());
    x = torch::sigmoid(fc2->forward(x));
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

int main(int argc [[maybe_unused]], char** argv [[maybe_unused]]) {
  at::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
  return 0;
}
