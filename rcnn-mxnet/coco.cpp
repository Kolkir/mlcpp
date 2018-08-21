#include "coco.h"

#include <rapidjson/error/en.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/reader.h>

#include <cstdlib>
#include <experimental/filesystem>
#include <iostream>
#include <string>

namespace fs = std::experimental::filesystem;

Coco::Coco(const std::string& path) {
  train_images_folder_ = fs::path(path) / "train2017";
  if (!fs::exists(train_images_folder_))
    throw std::runtime_error(train_images_folder_ + " folder missed");
  test_images_folder_ = fs::path(path) / "val2017";
  if (!fs::exists(test_images_folder_))
    throw std::runtime_error(test_images_folder_ + " folder missed");
  train_annotations_file_ =
      fs::path(path) / "annotations/instances_train2017.json";
  if (!fs::exists(train_annotations_file_))
    throw std::runtime_error(train_annotations_file_ + " file missed");
  test_annotations_file_ =
      fs::path(path) / "annotations/instances_val2017.json";
  if (!fs::exists(test_annotations_file_))
    throw std::runtime_error(test_annotations_file_ + " file missed");
}

struct CocoHandler
    : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, CocoHandler> {
  CocoHandler(Coco* coco) : coco_(coco) {}
  bool Double(double u) {
    if (annotation_object_) {
      if (bbox_array_) {
        annotation_.push_bbox(u);
      }
    }
    return true;
  }

  bool Uint(unsigned u) {
    if (image_object_) {
      if (key_ == "id") {
        image_.id = u;
      } else if (key_ == "width") {
        image_.width = u;
      } else if (key_ == "height") {
        image_.height = u;
      }
    } else if (category_object_) {
      if (key_ == "id") {
        category_.id = u;
      }
    } else if (annotation_object_) {
      if (key_ == "id") {
        annotation_.id = u;
      } else if (key_ == "image_id") {
        annotation_.image_id = u;
      } else if (key_ == "category_id") {
        annotation_.category_id = u;
      }
    }
    return true;
  }

  bool String(const char* str, rapidjson::SizeType length, bool /*copy*/) {
    if (image_object_) {
      if (key_ == "file_name") {
        image_.name.assign(str, length);
      }
    } else if (category_object_) {
      if (key_ == "name") {
        category_.name.assign(str, length);
      }
    }
    return true;
  }
  bool StartObject() {
    if (image_array_) {
      image_object_ = true;
    } else if (category_array_) {
      category_object_ = true;
    } else if (annotation_array_) {
      annotation_object_ = true;
    }
    return true;
  }
  bool Key(const char* str, rapidjson::SizeType length, bool /*copy*/) {
    key_ = std::string(str, length);
    return true;
  }
  bool EndObject(rapidjson::SizeType /*memberCount*/) {
    if (image_array_ && image_object_) {
      coco_->AddImage(image_);
      image_object_ = false;
    } else if (category_array_ && category_object_) {
      coco_->AddCategory(category_);
      category_object_ = false;
    } else if (annotation_array_ && annotation_object_) {
      coco_->AddAnnotation(annotation_);
      annotation_object_ = false;
    }
    key_.clear();
    return true;
  }
  bool StartArray() {
    if (key_ == "images") {
      image_array_ = true;
      array_close_ = &image_array_;
    } else if (key_ == "categories") {
      category_array_ = true;
      array_close_ = &category_array_;
    } else if (key_ == "annotations") {
      annotation_array_ = true;
      array_close_ = &annotation_array_;
    } else if (annotation_object_ && key_ == "bbox") {
      bbox_array_ = true;
      array_close_ = &bbox_array_;
    } else {
      array_close_ = nullptr;
    }
    return true;
  }
  bool EndArray(rapidjson::SizeType /*elementCount*/) {
    if (array_close_) {
      *array_close_ = false;
    }
    key_.clear();
    return true;
  }

  std::string key_;

  CocoAnnotation annotation_;
  bool annotation_array_ = false;
  bool annotation_object_ = false;

  CocoCategory category_;
  bool category_array_ = false;
  bool category_object_ = false;

  CocoImage image_;
  bool image_array_ = false;
  bool image_object_ = false;

  bool bbox_array_ = false;

  bool* array_close_ = nullptr;
  Coco* coco_ = nullptr;
};

void Coco::LoadTrainData() {
  auto* file = std::fopen(train_annotations_file_.c_str(), "r");
  if (file) {
    char readBuffer[65536];
    rapidjson::FileReadStream is(file, readBuffer, sizeof(readBuffer));
    rapidjson::Reader reader;
    CocoHandler handler(this);
    const uint32_t n_reserve = 100000;
    images_.reserve(n_reserve);
    annotations_.reserve(n_reserve);
    categories_.reserve(n_reserve);
    auto res = reader.Parse(is, handler);
    std::fclose(file);
    if (!res) {
      throw std::runtime_error(rapidjson::GetParseError_En(res.Code()));
    }
  } else {
    throw std::runtime_error(train_annotations_file_ + " file can't be opened");
  }
}

void Coco::AddImage(CocoImage image) {
  images_.emplace(std::make_pair(image.id, image));
}

void Coco::AddAnnotation(CocoAnnotation annotation) {
  annotations_.emplace(std::make_pair(annotation.id, annotation));
  image_to_ant_index_[annotation.image_id].insert(annotation.id);
}

void Coco::AddCategory(CocoCategory category) {
  categories_.emplace(std::make_pair(category.id, category));
}

uint32_t Coco::GetImagesCount() const {
  return static_cast<uint32_t>(images_.size());
}

ImageDesc Coco::GetImage(uint32_t index,
                         uint32_t short_side,
                         uint32_t long_side) const {
  if (index < images_.size()) {
    auto i = images_.begin();
    std::advance(i, index);
    fs::path file_path(train_images_folder_);
    file_path /= i->second.name;
    auto img = cv::imread(file_path.string());
    if (!img.empty()) {
      img.convertTo(img, CV_32FC3);
      // resize
      auto im_min_max = std::minmax(img.rows, img.cols);
      auto scale =
          static_cast<float>(short_side) / static_cast<float>(im_min_max.first);
      // prevent bigger axis from being more than max_size:
      if (std::round(scale * im_min_max.second) > long_side)
        scale = static_cast<float>(long_side) /
                static_cast<float>(im_min_max.second);
      cv::resize(img, img, cv::Size(), static_cast<double>(scale),
                 static_cast<double>(scale), cv::INTER_LINEAR);
      ImageDesc result;
      result.image = img;
      result.scale = scale;
      auto& ants = image_to_ant_index_.at(i->second.id);
      result.boxes.reserve(ants.size());
      result.classes.reserve(ants.size());
      for (const auto ant_id : ants) {
        const auto& ant = annotations_.at(ant_id);
        result.boxes.push_back(LabelBBox{static_cast<float>(ant.bbox.x),
                                         static_cast<float>(ant.bbox.y),
                                         static_cast<float>(ant.bbox.width),
                                         static_cast<float>(ant.bbox.height)});
        const auto& cat = categories_.at(ant.category_id);
        result.classes.push_back(static_cast<float>(cat.id));
      }
      return result;
    } else {
      throw std::runtime_error(file_path.string() + " file can't be opened");
    }
  } else {
    throw std::out_of_range("Image index is out of bounds");
  }
}

cv::Mat Coco::DrawAnnotedImage(uint32_t id) const {
  const auto& image = images_.at(id);
  fs::path file_path(train_images_folder_);
  file_path /= image.name;
  auto img = cv::imread(file_path.string());
  if (!img.empty()) {
    for (const auto ant_id : image_to_ant_index_.at(id)) {
      const auto& ant = annotations_.at(ant_id);
      cv::Point tl(static_cast<int32_t>(ant.bbox.x),
                   static_cast<int32_t>(ant.bbox.y));
      cv::Point br(tl.x + static_cast<int32_t>(ant.bbox.width),
                   tl.y + static_cast<int32_t>(ant.bbox.height));
      cv::rectangle(img, tl, br, cv::Scalar(255, 0, 0));
      const auto& cat = categories_.at(ant.category_id);
      cv::putText(img, cat.name, tl, cv::FONT_HERSHEY_PLAIN, 1,
                  cv::Scalar(0, 0, 255));
    }
    return img;
  } else {
    throw std::runtime_error(file_path.string() + " file can't be opened");
  }
}
