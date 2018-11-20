#include "imageutils.h"

#include <opencv2/opencv.hpp>

at::Tensor LoadImage(const std::string path) {
  // Idea taken from https://github.com/pytorch/pytorch/issues/12506

  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

  // we have to split the interleaved channels
  cv::Mat bgr[3];
  cv::split(image, bgr);
  cv::Mat channelsConcatenated;
  vconcat(bgr[0], bgr[1], channelsConcatenated);
  vconcat(channelsConcatenated, bgr[2], channelsConcatenated);

  cv::Mat channelsConcatenatedFloat;
  channelsConcatenated.convertTo(channelsConcatenatedFloat, CV_32FC3);

  std::vector<int64_t> dims{1, static_cast<int64_t>(image.channels()),
                            static_cast<int64_t>(image.rows),
                            static_cast<int64_t>(image.cols)};

  at::TensorOptions options(at::kFloat);
  at::Tensor tensor_image =
      torch::from_blob(channelsConcatenated.data, at::IntList(dims), options);
  return tensor_image;
}
