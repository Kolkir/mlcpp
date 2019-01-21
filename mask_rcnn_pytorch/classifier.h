#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <inttypes.h>

#include <torch/torch.h>

class ClassifierImpl : public torch::nn::Module {
 public:
  ClassifierImpl();
  ClassifierImpl(uint32_t depth,
                 uint32_t pool_size,
                 const std::vector<int32_t>& image_shape,
                 uint32_t num_classes);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      std::vector<torch::Tensor> feature_maps,
      torch::Tensor rois);

 private:
  torch::nn::Conv2d conv1_{nullptr};
  torch::nn::BatchNorm bn1_{nullptr};
  torch::nn::Conv2d conv2_{nullptr};
  torch::nn::BatchNorm bn2_{nullptr};
  torch::nn::Functional relu_{nullptr};
  torch::nn::Linear linear_class_{nullptr};
  torch::nn::Linear linear_bbox_{nullptr};

  uint32_t pool_size_{0};
  std::vector<int32_t> image_shape_;
};

TORCH_MODULE(Classifier);

#endif  // CLASSIFIER_H
