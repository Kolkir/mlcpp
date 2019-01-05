#ifndef COCODATASET_H
#define COCODATASET_H

#include "cocoloader.h"
#include "config.h"
#include "imageutils.h"

#include <torch/torch.h>

struct Sample {
  torch::Tensor image;
  ImageMeta image_meta;
  torch::Tensor rpn_match;
  torch::Tensor rpn_bbox;
  torch::Tensor gt_class_ids;
  torch::Tensor gt_boxes;
  torch::Tensor gt_masks;
};

class CocoDataset {
 public:
  CocoDataset(std::unique_ptr<CocoLoader> loader,
              std::shared_ptr<const Config> config);
  CocoDataset(const CocoDataset&) = delete;
  CocoDataset& operator=(const CocoDataset&) = delete;

  Sample Get(size_t index);
  size_t GetSize() const;

 private:
  std::unique_ptr<CocoLoader> loader_;
  std::shared_ptr<const Config> config_;
};

#endif  // COCODATASET_H
