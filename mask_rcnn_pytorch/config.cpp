#include "config.h"
#include "debug.h"

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

Config::Config() {
  auto cwd = fs::current_path();
  imagenet_model_path = cwd /= "resnet50_imagenet.pth";

  UpdateSettings();
}

void Config::UpdateSettings() {
  // Effective batch size
  if (gpu_count > 0) {
    batch_size = images_per_gpu * gpu_count;
  } else {
    batch_size = images_per_gpu;
  }

  // adjust step size based on batch size
  steps_per_epoch = batch_size * steps_per_epoch;

  // input image size
  image_shape = {image_max_dim, image_max_dim, 3};

  // compute backbone size from input image size
  backbone_shapes.clear();
  for (auto stride : backbone_strides) {
    backbone_shapes.push_back(
        {image_shape[0] / stride, image_shape[1] / stride});
  }
}
