#include "cocoloader.h"

#include "imageutils.h"

#include <rapidjson/error/en.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/reader.h>

#include <cstdlib>
#include <experimental/filesystem>
#include <iostream>
#include <string>

namespace fs = std::experimental::filesystem;

CocoLoader::CocoLoader(const std::string& images_folder,
                       const std::string& ann_file) {
  images_folder_ = fs::path(images_folder);
  if (!fs::exists(images_folder_))
    throw std::runtime_error(images_folder_ + " folder missed");
  annotations_file_ = fs::path(ann_file);
  if (!fs::exists(annotations_file_))
    throw std::runtime_error(annotations_file_ + " file missed");
}

struct CocoHandler
    : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, CocoHandler> {
  CocoHandler(CocoLoader* coco) : coco_(coco) {}
  bool Double(double u) {
    if (annotation_object_) {
      if (bbox_array_) {
        annotation_.push_bbox(u);
      } else if (segmentation_array_part_) {
        annotation_.push_segm_coord(u);
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
      } else if (key_ == "iscrowd") {
        annotation_.iscrowd = u == 1;
        if (annotation_.iscrowd) {
          std::cerr << "RLE segmentations are not supported\n";
          exit(0);
        }
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
    } else if (segmentation_array_) {
      std::cerr << "RLE segmentations are not supported\n";
      exit(0);
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
      annotation_.segmentation.clear();
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
    } else if (annotation_object_ && key_ == "segmentation" &&
               !segmentation_array_) {
      segmentation_array_ = true;
      array_close_ = &segmentation_array_;
    } else if (annotation_object_ && segmentation_array_) {
      segmentation_array_part_ = true;
      array_close_ = &segmentation_array_part_;
      annotation_.segmentation.emplace_back();
    } else {
      array_close_ = nullptr;
    }
    return true;
  }
  bool EndArray(rapidjson::SizeType /*elementCount*/) {
    // handle parent array
    if (annotation_object_ && segmentation_array_ &&
        !segmentation_array_part_) {
      segmentation_array_ = false;
    }

    if (array_close_) {
      *array_close_ = false;
    }
    key_.clear();
    return true;
  }

  std::string key_;

  CocoAnnotation annotation_;
  bool segmentation_array_ = false;
  bool annotation_array_ = false;
  bool annotation_object_ = false;

  CocoCategory category_;
  bool category_array_ = false;
  bool category_object_ = false;

  CocoImage image_;
  bool image_array_ = false;
  bool image_object_ = false;

  bool bbox_array_ = false;
  bool segmentation_array_part_ = false;

  bool* array_close_ = nullptr;
  CocoLoader* coco_ = nullptr;
};

void CocoLoader::LoadData(const std::vector<std::string>& coco_classes,
                          const std::vector<uint32_t>& keep_classes,
                          float keep_aspect) {
  if (!images_.empty()) {
    std::cerr << "Dataset " << images_folder_ << " already loaded\n";
    return;
  }
  auto* file = std::fopen(annotations_file_.c_str(), "r");
  if (file) {
    const uint32_t n_reserve = 100000;
    images_.reserve(n_reserve);
    annotations_.reserve(n_reserve);
    categories_.reserve(n_reserve);
    {  // force memory cleaning
      char readBuffer[65536];
      rapidjson::FileReadStream is(file, readBuffer, sizeof(readBuffer));
      rapidjson::Reader reader;
      CocoHandler handler(this);
      auto res = reader.Parse(is, handler);
      std::fclose(file);
      if (!res) {
        throw std::runtime_error(rapidjson::GetParseError_En(res.Code()));
      }
    }

    // remove images without annotations
    std::vector<uint32_t> images_to_remove;
    for (auto& img : images_) {
      auto image_id = img.first;
      auto i = image_to_ant_index_.find(image_id);
      if (i == image_to_ant_index_.end()) {
        images_to_remove.push_back(image_id);
      }
    }
    for (auto image_id : images_to_remove) {
      images_.erase(image_id);
    }

    // map categories to classes
    for (auto& cat : categories_) {
      auto i =
          std::find(coco_classes.begin(), coco_classes.end(), cat.second.name);
      auto pos = std::distance(coco_classes.begin(), i);
      cat_ind_to_class_ind_.insert({cat.second.id, static_cast<uint32_t>(pos)});
    }
    // filter - leave images with required aspect ration only
    if (keep_aspect > 0) {
      images_to_remove.clear();
      for (auto& img : images_) {
        auto aspect = static_cast<float>(img.second.width) /
                      static_cast<float>(img.second.height);
        if (std::abs(aspect - keep_aspect) > 0.001f) {
          images_to_remove.push_back(img.first);
        }
      }
      for (auto image_id : images_to_remove) {
        images_.erase(image_id);
      }
    }

    // filter - leave only required classes
    if (!keep_classes.empty()) {
      std::unordered_set<uint32_t> keep_classes_set;
      keep_classes_set.insert(keep_classes.begin(), keep_classes.end());
      images_to_remove.clear();
      for (auto& img : images_) {
        auto image_id = img.first;
        bool skip = true;
        std::vector<uint32_t> ant_to_remove;
        auto& ants = image_to_ant_index_[image_id];
        for (auto ant_id : ants) {
          auto class_ind =
              cat_ind_to_class_ind_[annotations_[ant_id].category_id];
          bool is_keep_class =
              keep_classes_set.find(class_ind) != keep_classes_set.end();
          if (is_keep_class) {
            skip = false;
          } else {
            ant_to_remove.push_back(ant_id);
          }
        }

        if (skip) {
          images_to_remove.push_back(image_id);
        } else {
          for (auto ant_id : ant_to_remove) {
            // leave only keep classes annotations
            ants.erase(ant_id);
          }
        }
      }
      for (auto image_id : images_to_remove) {
        images_.erase(image_id);
      }
    }
  } else {
    throw std::runtime_error(annotations_file_ + " file can't be opened");
  }
}

void CocoLoader::AddImage(CocoImage image) {
  images_.emplace(std::make_pair(image.id, image));
}

void CocoLoader::AddAnnotation(CocoAnnotation annotation) {
  if (annotation.bbox.height > 0 && annotation.bbox.width > 0 &&
      annotation.bbox.x >= 0 && annotation.bbox.y >= 0) {
    annotations_.emplace(std::make_pair(annotation.id, annotation));
    image_to_ant_index_[annotation.image_id].insert(annotation.id);
  }
}

void CocoLoader::AddCategory(CocoCategory category) {
  categories_.emplace(std::make_pair(category.id, category));
}

uint32_t CocoLoader::GetImagesCount() const {
  return static_cast<uint32_t>(images_.size());
}

static cv::Mat ConvertPolygonsToMask(
    const std::vector<std::vector<int32_t>>& polygons,
    const cv::Size& size) {
  cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);

  std::vector<std::vector<cv::Point>> contours(polygons.size());

  size_t c_idx = 0;
  for (const auto& poly : polygons) {
    auto len = poly.size() / 2;
    for (size_t i = 0; i < len; ++i) {
      auto p_idx = i * 2;
      contours[c_idx].push_back(cv::Point(poly[p_idx], poly[p_idx + 1]));
    }
    ++c_idx;
  }

  cv::drawContours(mask, contours, -1, cv::Scalar(255), cv::FILLED);

  return mask;
}

ImageDesc CocoLoader::GetImage(uint64_t index) const {
  if (index < images_.size()) {
    auto i = images_.begin();
    std::advance(i, index);
    fs::path file_path(images_folder_);
    file_path /= i->second.name;
    // std::cout << file_path << std::endl;
    cv::Mat img = LoadImage(file_path.string());
    if (!img.empty()) {
      ImageDesc result;
      result.id = i->second.id;
      result.image = img;
      auto& ants = image_to_ant_index_.at(i->second.id);
      result.boxes.reserve(ants.size());
      result.classes.reserve(ants.size());
      for (const auto ant_id : ants) {
        const auto& ant = annotations_.at(ant_id);
        result.boxes.push_back(ant.bbox);
        const auto& cat = categories_.at(ant.category_id);
        uint32_t class_ind = cat_ind_to_class_ind_.at(cat.id);
        result.classes.push_back(static_cast<int32_t>(class_ind));

        result.masks.push_back(
            ConvertPolygonsToMask(ant.segmentation, img.size()));
      }
      return result;
    } else {
      throw std::runtime_error(file_path.string() + " file can't be opened");
    }
  } else {
    throw std::out_of_range("Image index is out of bounds");
  }
}

cv::Mat CocoLoader::DrawAnnotedImage(uint32_t id) const {
  const auto& image = images_.at(id);
  fs::path file_path(images_folder_);
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

      cv::Mat mask_ch[3];
      mask_ch[2] = ConvertPolygonsToMask(ant.segmentation, img.size());
      mask_ch[0] = cv::Mat::zeros(img.size(), CV_8UC1);
      mask_ch[1] = cv::Mat::zeros(img.size(), CV_8UC1);
      cv::Mat mask;
      cv::merge(mask_ch, 3, mask);
      cv::addWeighted(img, 1, mask, 0.5, 0, img);

      const auto& cat = categories_.at(ant.category_id);
      cv::putText(img, cat.name, tl, cv::FONT_HERSHEY_PLAIN, 1,
                  cv::Scalar(0, 0, 255));
    }
    return img;
  } else {
    throw std::runtime_error(file_path.string() + " file can't be opened");
  }
}
