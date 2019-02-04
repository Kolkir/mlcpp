#include "visualize.h"
#include "debug.h"

#include <string>
#include <vector>

void visualize(const cv::Mat& image,
               at::Tensor boxes,
               at::Tensor class_ids,
               at::Tensor scores,
               const std::vector<cv::Mat>& masks,
               float score_threshold,
               const std::vector<std::string>& class_names) {
  cv::Mat img = image.clone();
  auto n = boxes.size(0);
  for (int64_t i = 0; i < n; ++i) {
    auto score = *scores[i].data<float>();
    if (score >= score_threshold) {
      auto bbox = boxes[i];
      auto y1 = *bbox[0].data<int32_t>();
      auto x1 = *bbox[1].data<int32_t>();
      auto y2 = *bbox[2].data<int32_t>();
      auto x2 = *bbox[3].data<int32_t>();
      auto class_id = *class_ids[i].data<int64_t>();

      cv::Mat bin_mask = masks[i].clone();
      cv::Mat mask_ch[3];
      mask_ch[2] = bin_mask;
      mask_ch[0] = cv::Mat::zeros(img.size(), CV_8UC1);
      mask_ch[1] = cv::Mat::zeros(img.size(), CV_8UC1);
      cv::Mat mask;
      cv::merge(mask_ch, 3, mask);
      cv::addWeighted(img, 1, mask, 0.5, 0, img);

      cv::Point tl(static_cast<int>(x1), static_cast<int>(y1));
      cv::Point br(static_cast<int>(x2), static_cast<int>(y2));
      cv::rectangle(img, tl, br, cv::Scalar(1, 0, 0));
      cv::putText(img,
                  class_names[static_cast<size_t>(class_id)] + " - " +
                      std::to_string(score),
                  cv::Point(tl.x + 5, tl.y + 5),   // Coordinates
                  cv::FONT_HERSHEY_COMPLEX_SMALL,  // Font
                  1.0,                             // Scale. 2.0 = 2x bigger
                  cv::Scalar(255, 100, 255));      // BGR Color
    }
  }
  cv::imwrite("result.png", img);
}
