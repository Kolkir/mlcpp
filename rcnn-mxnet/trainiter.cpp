#include "trainiter.h"
#include "imageutils.h"

#include <mshadow/tensor.h>
#include <Eigen/Dense>

#include <algorithm>

// uncomment to save batch images with bboxes
// #define IMG_DEBUG_TEST

TrainIter::TrainIter(ImageDb* image_db,
                     const Params& params,
                     uint32_t feat_height,
                     uint32_t feat_width)
    : image_db_(image_db),
      batch_size_(params.rcnn_batch_size),
      short_side_len_(params.img_short_side),
      long_side_len_(params.img_long_side),
      one_image_size_(3 * short_side_len_ * long_side_len_),
      num_anchors_(static_cast<uint32_t>(params.rpn_anchor_scales.size() *
                                         params.rpn_anchor_ratios.size())),
      batch_gt_boxes_count_(params.rcnn_batch_gt_boxes),
      feat_height_(feat_height),
      feat_width_(feat_width),
      batch_indices_(batch_size_),
      anchor_generator_(params),
      anchor_sampler_(params) {
  assert(image_db_ != nullptr);
  size_ = image_db->GetImagesCount();
  Reset();
}

uint32_t TrainIter::GetSize() const {
  return size_;
}

uint32_t TrainIter::GetBatchCount() const {
  return size_ / batch_size_;
}

void TrainIter::Reset() {
  cur_ = 0;
  data_indices_.resize(size_);
  std::iota(data_indices_.begin(), data_indices_.end(), 0);
  std::shuffle(data_indices_.begin(), data_indices_.end(), random_engine_);
}

bool TrainIter::Next() {
  if (cur_ + batch_size_ <= size_) {
    auto s = data_indices_.begin() + cur_;
    auto e = s + batch_size_;
    std::copy(s, e, batch_indices_.begin());

    FillData();
    FillLabels();

    cur_ += batch_size_;
    return true;
  } else {
    return false;
  }
}

void TrainIter::GetData(mxnet::cpp::NDArray& im_arr,
                        mxnet::cpp::NDArray& im_info_arr,
                        mxnet::cpp::NDArray& gt_boxes_arr,
                        mxnet::cpp::NDArray& label_arr,
                        mxnet::cpp::NDArray& bbox_target_arr,
                        mxnet::cpp::NDArray& bbox_weight_arr) {
  im_arr.SyncCopyFromCPU(raw_im_data_.data(), raw_im_data_.size());
  im_arr.WaitAll();
  im_info_arr.SyncCopyFromCPU(raw_im_info_data_.data(),
                              raw_im_info_data_.size());
  im_info_arr.WaitAll();
  gt_boxes_arr.SyncCopyFromCPU(raw_gt_boxes_data_.data(),
                               raw_gt_boxes_data_.size());
  gt_boxes_arr.WaitAll();
  label_arr.SyncCopyFromCPU(raw_label_.data(), raw_label_.size());
  label_arr.WaitAll();
  bbox_target_arr.SyncCopyFromCPU(raw_bbox_target_.data(),
                                  raw_bbox_target_.size());
  bbox_target_arr.WaitAll();
  bbox_weight_arr.SyncCopyFromCPU(raw_bbox_weight_.data(),
                                  raw_bbox_weight_.size());
  bbox_weight_arr.WaitAll();
}

void TrainIter::FillData() {
  raw_gt_boxes_data_.clear();
  raw_im_data_.clear();
  raw_im_info_data_.clear();

  auto if_i = std::back_insert_iterator<std::vector<float>>(raw_im_info_data_);
  auto b_i = std::back_insert_iterator<std::vector<float>>(raw_gt_boxes_data_);

  // gt_boxes padding markers
  std::vector<std::pair<size_t, size_t>> gt_boxes_pad_markers;

  for (auto index : batch_indices_) {
    // image is loaded with padding
    auto image_desc =
        image_db_->GetImage(index, short_side_len_, long_side_len_);
    // Fill image
    auto array = CVToMxnetFormat(image_desc.image);
    assert(array.size() <= one_image_size_);
    raw_im_data_.insert(raw_im_data_.end(), array.begin(), array.end());

    // Fill info
    *if_i++ = image_desc.height;
    *if_i++ = image_desc.width;
    *if_i++ = image_desc.scale;
    // Pad is not required

    // Fill boxes
    if (image_desc.boxes.size() > batch_gt_boxes_count_)
      image_desc.boxes.resize(batch_gt_boxes_count_);
#ifdef IMG_DEBUG_TEST
    cv::Mat imgCopy = image_desc.image.clone();
#endif
    auto ic = image_desc.classes.begin();
    for (const auto& b : image_desc.boxes) {
      // sanitize box
      auto x1 = std::max(0.f, b.x * image_desc.scale);
      auto y1 = std::max(0.f, b.y * image_desc.scale);
      auto x2 = std::min(image_desc.width - 1,
                         x1 + std::max(0.f, b.width * image_desc.scale - 1));
      auto y2 = std::min(image_desc.height - 1,
                         y1 + std::max(0.f, b.height * image_desc.scale - 1));
      *b_i++ = std::trunc(x1);
      *b_i++ = std::trunc(y1);
      *b_i++ = std::trunc(x2);
      *b_i++ = std::trunc(y2);

      auto class_index = *(ic++);
      *b_i++ = class_index;  // class index

#ifdef IMG_DEBUG_TEST
      cv::Point tl(static_cast<int>(x1), static_cast<int>(y1));
      cv::Point br(static_cast<int>(x2), static_cast<int>(y2));
      cv::rectangle(imgCopy, tl, br, cv::Scalar(100, 100, 255));
      cv::putText(imgCopy, std::to_string(class_index),
                  cv::Point(tl.x + 5, tl.y + 5),   // Coordinates
                  cv::FONT_HERSHEY_COMPLEX_SMALL,  // Font
                  1.0,                             // Scale. 2.0 = 2x bigger
                  cv::Scalar(100, 100, 255));      // BGR Color
#endif
    }
#ifdef IMG_DEBUG_TEST
    cv::imwrite("det.png", imgCopy);
#endif
    gt_boxes_pad_markers.push_back(
        {raw_gt_boxes_data_.size(), image_desc.boxes.size()});
  }

  // Add padding to gt_boxes
  size_t pad_pos = 0;
  for (auto& pad_marker : gt_boxes_pad_markers) {
    auto pad_size = batch_gt_boxes_count_ - pad_marker.second;
    if (pad_size != 0) {
      auto pos = raw_gt_boxes_data_.begin();
      std::advance(pos, pad_marker.first + pad_pos);
      raw_gt_boxes_data_.insert(pos, pad_size * 5, -1.f);
      pad_pos += pad_size * 5;
    }
  }
}

void TrainIter::FillLabels() {
  // all stacked image share same anchors
  auto anchors = anchor_generator_.Generate(feat_width_, feat_height_);
#ifdef IMG_DEBUG_TEST
  cv::Mat img = cv::imread("det.png");
  for (Eigen::Index i = 0; i < anchors.rows(); ++i) {
    cv::Point tl(static_cast<int>(anchors(i, 0)),
                 static_cast<int>(anchors(i, 1)));
    cv::Point br(static_cast<int>(anchors(i, 2)),
                 static_cast<int>(anchors(i, 3)));
    cv::rectangle(img, tl, br, cv::Scalar(255, 100, 100));
  }
  cv::imwrite("det.png", img);
#endif

  // prepare data bindings
  raw_label_.resize(static_cast<size_t>(anchors.rows()) * batch_size_);
  Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      label_map(raw_label_.data(), anchors.rows() * batch_size_, 1);

  raw_bbox_target_.resize(
      static_cast<size_t>(anchors.rows() * 4 * batch_size_));
  Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      bbox_target_map(raw_bbox_target_.data(), anchors.rows() * batch_size_, 4);

  raw_bbox_weight_.resize(
      static_cast<size_t>(anchors.rows() * 4 * batch_size_));
  Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      bbox_weight_map(raw_bbox_weight_.data(), anchors.rows() * batch_size_, 4);

  // assign anchor according to their real size encoded in im_info
  auto all_boxes = Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      raw_gt_boxes_data_.data(),
      static_cast<mx_uint>(raw_gt_boxes_data_.size() / 5), 5);
  for (uint32_t i = 0, box_index = 0; i < batch_size_; ++i) {
    auto im_width = raw_im_info_data_[i * 3 + 1];
    auto im_height = raw_im_info_data_[i * 3];
    auto boxes = all_boxes.block(box_index, 0, batch_gt_boxes_count_, 4);
    box_index += batch_gt_boxes_count_;

    Eigen::MatrixXf b_label, b_bbox_target, b_bbox_weight;
    std::tie(b_label, b_bbox_target, b_bbox_weight) =
        anchor_sampler_.Assign(anchors, boxes, im_width, im_height);

    // Because we use fixed image size padding is not required - number of valid
    // anchors will be the same

    label_map.block(i * anchors.rows(), 0, anchors.rows(), 1) = b_label;

    bbox_target_map.block(i * anchors.rows(), 0, anchors.rows(), 4) =
        b_bbox_target;
    bbox_weight_map.block(i * anchors.rows(), 0, anchors.rows(), 4) =
        b_bbox_weight;
  }

  // fix sizes
  ReshapeLabels(static_cast<unsigned int>(label_map.rows()),
                static_cast<unsigned int>(label_map.cols()));
  ReshapeTargets(static_cast<unsigned int>(bbox_target_map.rows()),
                 static_cast<unsigned int>(bbox_target_map.cols()));
  ReshapeWeights(static_cast<unsigned int>(bbox_weight_map.rows()),
                 static_cast<unsigned int>(bbox_weight_map.cols()));
}

void TrainIter::ReshapeLabels(unsigned int rows, unsigned int cols) {
  using namespace mshadow;
  try {
    auto out_shape =
        Shape4(batch_size_, 1, num_anchors_ * feat_height_, feat_width_);
    TensorContainer<cpu, 4, float> x2(out_shape);
    {
      Tensor<cpu, 2, float> lables(raw_label_.data(), Shape2(rows, cols));
      auto x1 = expr::reshape(
          lables,
          Shape4(batch_size_, num_anchors_ * feat_height_, feat_width_, 1));
      x2 = expr::transpose(x1, Shape4(0, 3, 1, 2));
    }

    // self test
    // std::fill(raw_label_.begin(), raw_label_.end(), 0);

    Tensor<cpu, 4, float> out_lables(raw_label_.data(), out_shape);
    Copy(out_lables, x2);
  } catch (const std::exception& ex) {
    std::cout << ex.what() << std::endl;
  }
}
void TrainIter::ReshapeTargets(unsigned int rows, unsigned int cols) {
  using namespace mshadow;
  try {
    auto out_shape =
        Shape4(batch_size_, 4 * num_anchors_, feat_height_, feat_width_);
    TensorContainer<cpu, 4, float> x2(out_shape);
    {
      Tensor<cpu, 2, float> bbox_targets(raw_bbox_target_.data(),
                                         Shape2(rows, cols));
      auto x1 = expr::reshape(
          bbox_targets,
          Shape4(batch_size_, feat_height_, feat_width_, 4 * num_anchors_));
      x2 = expr::transpose(x1, Shape4(0, 3, 1, 2));
    }

    // self test
    // std::fill(raw_bbox_target_.begin(), raw_bbox_target_.end(), 0);

    Tensor<cpu, 4, float> bbox_targets(raw_bbox_target_.data(), out_shape);
    Copy(bbox_targets, x2);
  } catch (const std::exception& ex) {
    std::cout << ex.what() << std::endl;
  }
}
void TrainIter::ReshapeWeights(unsigned int rows, unsigned int cols) {
  using namespace mshadow;
  try {
    auto out_shape =
        Shape4(batch_size_, 4 * num_anchors_, feat_height_, feat_width_);
    TensorContainer<cpu, 4, float> x2(out_shape);
    {
      Tensor<cpu, 2, float> bbox_weights(raw_bbox_weight_.data(),
                                         Shape2(rows, cols));
      auto x1 = expr::reshape(
          bbox_weights,
          Shape4(batch_size_, feat_height_, feat_width_, 4 * num_anchors_));
      x2 = expr::transpose(x1, Shape4(0, 3, 1, 2));
    }

    // self test
    // std::fill(raw_bbox_target_.begin(), raw_bbox_target_.end(), 0);

    Tensor<cpu, 4, float> bbox_weights(raw_bbox_weight_.data(), out_shape);
    Copy(bbox_weights, x2);
  } catch (const std::exception& ex) {
    std::cout << ex.what() << std::endl;
  }
}
