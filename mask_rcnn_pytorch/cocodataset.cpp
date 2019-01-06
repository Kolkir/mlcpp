#include "cocodataset.h"

CocoDataset::CocoDataset(std::shared_ptr<CocoLoader> loader,
                         std::shared_ptr<const Config> config)
    : loader_(loader), config_(config) {
  loader_->LoadData();
}

Sample CocoDataset::get(size_t index) {
  auto img_desc = loader_->GetImage(index);

  auto [image, window, scale, padding] =
      ResizeImage(img_desc.image, config_->image_min_dim,
                  config_->image_max_dim, config_->image_padding);

  auto masks = ResizeMasks(img_desc.masks, scale, padding);

  // TODO: resize boxes

  Sample result;
  return result;
}

torch::optional<size_t> CocoDataset::size() const {
  return loader_->GetImagesCount();
}
