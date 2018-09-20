#include "imageutils.h"

std::pair<cv::Mat, float> LoadImage(const std::string& file_name,
                                    uint32_t short_side,
                                    uint32_t long_side) {
  auto img = cv::imread(file_name);
  std::cout << "Load file " << file_name << std::endl;
  if (!img.empty()) {
    img.convertTo(img, CV_32FC3);
    float scale = 0;
    // resize

    auto im_min_max = std::minmax(img.rows, img.cols);
    scale =
        static_cast<float>(short_side) / static_cast<float>(im_min_max.first);
    // prevent bigger axis from being more than max_size:
    if (std::round(scale * im_min_max.second) > long_side) {
      scale =
          static_cast<float>(long_side) / static_cast<float>(im_min_max.second);
    }
    cv::resize(img, img, cv::Size(), static_cast<double>(scale),
               static_cast<double>(scale), cv::INTER_LINEAR);

    return {img, scale};
  }
  return {cv::Mat(), 0};
}

std::tuple<cv::Mat, float> LoadImageFitSize(const std::string& file_name,
                                            uint32_t height,
                                            uint32_t width) {
  auto img = cv::imread(file_name);
  std::cout << "Load file " << file_name << std::endl;
  if (!img.empty()) {
    img.convertTo(img, CV_32FC3);
    float scale = 1.f;
    // assume that an image width in most cases is bigger than height
    float ratio = static_cast<float>(img.cols) / static_cast<float>(img.rows);
    auto new_height = static_cast<int>(width / ratio);
    if (new_height <= static_cast<int>(height)) {
      scale = static_cast<float>(new_height) / static_cast<float>(img.rows);
    } else {
      ratio = static_cast<float>(img.rows) / static_cast<float>(img.cols);
      auto new_width = static_cast<int>(height / ratio);
      assert(new_width <= static_cast<int>(width));
      scale = static_cast<float>(new_width) / static_cast<float>(img.cols);
    }

    // resize
    cv::resize(img, img, cv::Size(), static_cast<double>(scale),
               static_cast<double>(scale),
               scale >= 1.f ? cv::INTER_LINEAR : cv::INTER_AREA);

    // pad
    int bottom = static_cast<int>(height) - img.rows;
    int right = static_cast<int>(width) - img.cols;
    if (right < 0 || bottom < 0) {
      std::cout << "error\n";
    }
    cv::copyMakeBorder(img, img, 0, bottom, 0, right, cv::BORDER_CONSTANT,
                       cv::Scalar(0, 0, 0, 0));

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
