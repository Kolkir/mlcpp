#ifndef COCODATASET_H
#define COCODATASET_H

#include "cocoloader.h"

#include <torch/torch.h>

class CocoDataset : public torch::data::Dataset<CocoDataset> {
 public:
  CocoDataset(std::unique_ptr<CocoLoader> loader);
  CocoDataset(const CocoDataset&) = delete;
  CocoDataset& operator=(const CocoDataset&) = delete;

  ExampleType get(size_t index) override;
  torch::optional<size_t> size() const override;

 private:
  std::unique_ptr<CocoLoader> loader_;
};

#endif  // COCODATASET_H
