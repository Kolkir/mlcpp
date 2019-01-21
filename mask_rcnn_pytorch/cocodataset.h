#ifndef COCODATASET_H
#define COCODATASET_H

#include "cocoloader.h"
#include "config.h"
#include "imageutils.h"

#include <torch/torch.h>

struct Input {
  torch::Tensor image;
  ImageMeta image_meta;
};

struct Target {
  torch::Tensor rpn_match;
  torch::Tensor rpn_bbox;
  torch::Tensor gt_class_ids;
  torch::Tensor gt_boxes;
  torch::Tensor gt_masks;
};

using Sample = torch::data::Example<Input, Target>;

class CocoDataset : public torch::data::Dataset<CocoDataset, Sample> {
 public:
  CocoDataset(std::shared_ptr<CocoLoader> loader,
              std::shared_ptr<const Config> config);

  Sample get(size_t index) override;
  torch::optional<size_t> size() const override;

 private:
  std::shared_ptr<CocoLoader> loader_;
  std::shared_ptr<const Config> config_;
  torch::Tensor anchors_;
};

#endif  // COCODATASET_H
