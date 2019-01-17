#include "visualize.h"
#include "debug.h"

#include <string>
#include <vector>

// COCO Class names
// Index of the class in the list is its ID. For example, to get ID of
// the teddy bear class, use: class_names.index("teddy bear")
static std::vector<std::string> class_names = {"BG",
                                               "person",
                                               "bicycle",
                                               "car",
                                               "motorcycle",
                                               "airplane",
                                               "bus",
                                               "train",
                                               "truck",
                                               "boat",
                                               "traffic light",
                                               "fire hydrant",
                                               "stop sign",
                                               "parking meter",
                                               "bench",
                                               "bird",
                                               "cat",
                                               "dog",
                                               "horse",
                                               "sheep",
                                               "cow",
                                               "elephant",
                                               "bear",
                                               "zebra",
                                               "giraffe",
                                               "backpack",
                                               "umbrella",
                                               "handbag",
                                               "tie",
                                               "suitcase",
                                               "frisbee",
                                               "skis",
                                               "snowboard",
                                               "sports ball",
                                               "kite",
                                               "baseball bat",
                                               "baseball glove",
                                               "skateboard",
                                               "surfboard",
                                               "tennis racket",
                                               "bottle",
                                               "wine glass",
                                               "cup",
                                               "fork",
                                               "knife",
                                               "spoon",
                                               "bowl",
                                               "banana",
                                               "apple",
                                               "sandwich",
                                               "orange",
                                               "broccoli",
                                               "carrot",
                                               "hot dog",
                                               "pizza",
                                               "donut",
                                               "cake",
                                               "chair",
                                               "couch",
                                               "potted plant",
                                               "bed",
                                               "dining table",
                                               "toilet",
                                               "tv",
                                               "laptop",
                                               "mouse",
                                               "remote",
                                               "keyboard",
                                               "cell phone",
                                               "microwave",
                                               "oven",
                                               "toaster",
                                               "sink",
                                               "refrigerator",
                                               "book",
                                               "clock",
                                               "vase",
                                               "scissors",
                                               "teddy bear",
                                               "hair drier",
                                               "toothbrush"};

void visualize(const cv::Mat& image,
               at::Tensor boxes,
               at::Tensor class_ids,
               at::Tensor scores,
               const std::vector<cv::Mat>& masks) {
  cv::Mat img = image.clone();
  auto n = boxes.size(0);
  for (int64_t i = 0; i < n; ++i) {
    auto bbox = boxes[i];
    auto y1 = *bbox[0].data<int32_t>();
    auto x1 = *bbox[1].data<int32_t>();
    auto y2 = *bbox[2].data<int32_t>();
    auto x2 = *bbox[3].data<int32_t>();
    auto class_id = *class_ids[i].data<int64_t>();
    auto score = *scores[i].data<float>();

    cv::Mat bin_mask = masks[i].clone();
    bin_mask = bin_mask != 0;
    bin_mask *= 255;
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
  cv::imwrite("result.png", img);
}
