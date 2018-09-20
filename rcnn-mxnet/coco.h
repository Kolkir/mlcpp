#ifndef COCO_H
#define COCO_H

#include "imagedb.h"

#include <opencv2/opencv.hpp>

#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

struct CocoCategory {
  uint32_t id = 0;
  std::string name;
};

struct CocoBBox {
  double x = 0;
  double y = 0;
  double width = 0;
  double height = 0;
};

struct CocoAnnotation {
  uint32_t id = 0;
  uint32_t image_id = 0;
  uint32_t category_id = 0;
  CocoBBox bbox;  // x,y,w,h
  void push_bbox(double v) {
    switch (bbox_index) {
      case 0:
        bbox.x = v;
        break;
      case 1:
        bbox.y = v;
        break;
      case 2:
        bbox.width = v;
        break;
      case 3:
        bbox.height = v;
        break;
    }
    bbox_index += 1;
    if (bbox_index >= 4) {
      bbox_index = 0;
    }
  }
  uint32_t bbox_index = 0;
};

struct CocoImage {
  uint32_t id = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  std::string name;
};

class Coco : public ImageDb {
 public:
  explicit Coco(const std::string& path);
  void LoadTrainData();
  void AddImage(CocoImage image);
  void AddAnnotation(CocoAnnotation annotation);
  void AddCategory(CocoCategory category);
  cv::Mat DrawAnnotedImage(uint32_t id) const;

  static const std::vector<std::string> GetClasses();

  // ImageDb interface
  uint32_t GetImagesCount() const override;
  ImageDesc GetImage(uint32_t index,
                     uint32_t height,
                     uint32_t width) const override;

 private:
  std::string train_images_folder_;
  std::string test_images_folder_;
  std::string train_annotations_file_;
  std::string test_annotations_file_;

  std::unordered_map<uint32_t, CocoImage> images_;
  std::unordered_map<uint32_t, CocoAnnotation> annotations_;
  std::unordered_map<uint32_t, CocoCategory> categories_;

  std::unordered_map<uint32_t, std::unordered_set<uint32_t>>
      image_to_ant_index_;
  std::unordered_map<uint32_t, uint32_t> cat_ind_to_class_ind_;
};

#endif  // COCO_H
