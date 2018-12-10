#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H

#include "config.h"

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

cv::Mat LoadImage(const std::string path);

at::Tensor CvImageToTensor(const cv::Mat& image);

struct Window {
  int32_t y1{0};
  int32_t x1{0};
  int32_t y2{0};
  int32_t x2{0};
};

struct Padding {
  int32_t top_pad{0};
  int32_t bottom_pad{0};
  int32_t left_pad{0};
  int32_t right_pad{0};
  int32_t front_pad{0};
  int32_t rear_pad{0};
};

struct ImageMeta {
  int32_t image_id{0};
  int32_t image_width{0};
  int32_t image_height{0};
  Window window;
  std::vector<int32_t> active_class_ids;
};

std::tuple<cv::Mat, Window, float, Padding> ResizeImage(
    cv::Mat image,
    int32_t min_dim,
    int32_t max_dim,
    bool do_padding = false);

std::tuple<at::Tensor, std::vector<ImageMeta>, std::vector<Window>> MoldInputs(
    const std::vector<cv::Mat>& images,
    const Config& config);

#endif  // IMAGEUTILS_H
