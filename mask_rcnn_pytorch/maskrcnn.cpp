#include "maskrcnn.h"

MaskRCNN::MaskRCNN(std::string model_dir, std::shared_ptr<Config const> config)
    : model_dir_(model_dir), config_(config) {
  Build();
  InitializeWeights();
}

bool MaskRCNN::Detect(const at::Tensor& image) {
  return false;
}

// Build Mask R-CNN architecture.
void MaskRCNN::Build() {}

void MaskRCNN::InitializeWeights() {}
