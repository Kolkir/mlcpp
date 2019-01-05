#ifndef MASKRCNN_H
#define MASKRCNN_H

#include "classifier.h"
#include "cocodataset.h"
#include "config.h"
#include "fpn.h"
#include "imageutils.h"
#include "mask.h"
#include "rpn.h"

#include <torch/torch.h>
#include <memory>

class MaskRCNNImpl : public torch::nn::Module {
 public:
  MaskRCNNImpl(std::string model_dir, std::shared_ptr<Config const> config);

  std::tuple<at::Tensor, at::Tensor> Detect(
      at::Tensor images,
      const std::vector<ImageMeta>& image_metas);

  void Train(std::unique_ptr<CocoDataset> train_dataset,
             std::unique_ptr<CocoDataset> val_dataset,
             float learning_rate,
             uint32_t epochs,
             const std::string& layers);

 private:
  void Build();
  void InitializeWeights();

  enum class Mode { Inference, Training };

  std::tuple<at::Tensor, at::Tensor> Predict(
      at::Tensor images,
      const std::vector<ImageMeta>& image_metas,
      Mode mode);

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
