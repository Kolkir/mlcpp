#include "maskrcnn.h"

MaskRCNN::MaskRCNN(std::string model_dir, std::shared_ptr<Config const> config)
    : model_dir_(model_dir), config_(std::move(config)) {}

bool MaskRCNN::detect(const at::Tensor& image) {}
