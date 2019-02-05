#ifndef COCO_H
#define COCO_H

#include <torch/torch.h>

#include <opencv2/opencv.hpp>

#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

struct CocoCategory {
  uint32_t id{0};
  std::string name;
};

struct CocoBBox {
  int32_t x{0};
  int32_t y{0};
  int32_t width{0};
  int32_t height{0};
};

struct CocoAnnotation {
  std::vector<std::vector<int32_t>> segmentation;
  CocoBBox bbox;  // x,y,w,h
  uint32_t id{0};
  uint32_t image_id{0};
  uint32_t category_id{0};
  uint32_t bbox_index{0};
  bool iscrowd{false};

  void push_bbox(double v) {
    switch (bbox_index) {
      case 0:
        bbox.x = static_cast<int32_t>(v);
        break;
      case 1:
        bbox.y = static_cast<int32_t>(v);
        break;
      case 2:
        bbox.width = static_cast<int32_t>(v);
        break;
      case 3:
        bbox.height = static_cast<int32_t>(v);
        break;
    }
    bbox_index += 1;
    if (bbox_index >= 4) {
      bbox_index = 0;
    }
  }
  void push_segm_coord(double v) {
    segmentation.back().push_back(static_cast<int32_t>(v));
  }
};

struct CocoImage {
  uint32_t id{0};
  uint32_t width{0};
  uint32_t height{0};
  std::string name;
};

struct ImageDesc {
  uint32_t id{0};
  cv::Mat image;
  std::vector<cv::Mat> masks;
  std::vector<CocoBBox> boxes;
  std::vector<int32_t> classes;
};

class CocoLoader {
 public:
  explicit CocoLoader(const std::string& images_folder,
                      const std::string& ann_file);
  void LoadData(const std::vector<std::string>& coco_classes,
                const std::vector<uint32_t>& keep_classes = {},
                float keep_aspect = -1);
  void AddImage(CocoImage image);
  void AddAnnotation(CocoAnnotation annotation);
  void AddCategory(CocoCategory category);
  cv::Mat DrawAnnotedImage(uint32_t id) const;

  // ImageDb interface
  uint32_t GetImagesCount() const;
  ImageDesc GetImage(uint64_t index) const;

 private:
  std::string images_folder_;
  std::string annotations_file_;

  std::unordered_map<uint32_t, CocoImage> images_;
  std::unordered_map<uint32_t, CocoAnnotation> annotations_;
  std::unordered_map<uint32_t, CocoCategory> categories_;

  std::unordered_map<uint32_t, std::unordered_set<uint32_t>>
      image_to_ant_index_;
  std::unordered_map<uint32_t, uint32_t> cat_ind_to_class_ind_;
};

#endif  // COCO_H
