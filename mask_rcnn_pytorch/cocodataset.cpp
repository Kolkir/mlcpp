#include "cocodataset.h"

CocoDataset::CocoDataset(std::unique_ptr<CocoLoader> loader)
    : loader_(std::move(loader)) {
  loader_->LoadData();
}

CocoDataset::ExampleType CocoDataset::get(size_t index) {
  auto img_desc = loader_->GetImage(index);

  CocoDataset::ExampleType result;
  return result;
}

torch::optional<size_t> CocoDataset::size() const {
  return loader_->GetImagesCount();
}
