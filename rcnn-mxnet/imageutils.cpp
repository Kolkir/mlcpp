#include "imageutils.h"

std::pair<cv::Mat, float> LoadImage(const std::string& file_name,
                                    uint32_t short_side,
                                    uint32_t long_side,
                                    bool force_size) {
  auto img = cv::imread(file_name);
  if (!img.empty()) {
    img.convertTo(img, CV_32FC3);
    float scale = 0;
    // resize
    if (force_size) {
      cv::resize(
          img, img,
          cv::Size(static_cast<int>(short_side), static_cast<int>(long_side)),
          0, 0, cv::INTER_LINEAR);
    } else {
      auto im_min_max = std::minmax(img.rows, img.cols);
      scale =
          static_cast<float>(short_side) / static_cast<float>(im_min_max.first);
      // prevent bigger axis from being more than max_size:
      if (std::round(scale * im_min_max.second) > long_side) {
        scale = static_cast<float>(long_side) /
                static_cast<float>(im_min_max.second);
      }
      cv::resize(img, img, cv::Size(), static_cast<double>(scale),
                 static_cast<double>(scale), cv::INTER_LINEAR);
    }
    return {img, scale};
  }
  return {cv::Mat(), 0};
}

std::vector<float> CVToMxnetFormat(const cv::Mat& img) {
  assert(img.type() == CV_32FC3);
  auto size = img.channels() * img.rows * img.cols;
  std::vector<float> array(static_cast<size_t>(size));
  size_t k = 0;
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < img.rows; ++y) {
      for (int x = 0; x < img.cols; ++x) {
        auto t = (y * img.cols + x) * 3 + c;
        array[k] = img.ptr<float>()[t];
        ++k;
      }
    }
  }
  return array;
}

void ShowResult(const std::vector<Detection>& detection,
                const std::string& file_name,
                const std::string& out_file_name,
                const std::vector<std::string>& classes,
                float thresh) {
  auto img = cv::imread(file_name);
  if (!img.empty()) {
    img.convertTo(img, CV_32FC3);
    for (auto& det : detection) {
      if (det.class_id > 0 && det.score > thresh) {
        cv::Point tl(static_cast<int>(det.x1), static_cast<int>(det.y1));
        cv::Point br(static_cast<int>(det.x2), static_cast<int>(det.y2));
        cv::rectangle(img, tl, br, cv::Scalar(1, 0, 0));
        cv::putText(img,
                    classes[static_cast<size_t>(det.class_id)] + " - " +
                        std::to_string(det.score),
                    cv::Point(tl.x + 5, tl.y + 5),   // Coordinates
                    cv::FONT_HERSHEY_COMPLEX_SMALL,  // Font
                    1.0,                             // Scale. 2.0 = 2x bigger
                    cv::Scalar(100, 100, 255));      // BGR Color
      }
    }
    cv::imwrite(out_file_name, img);
  }
}
