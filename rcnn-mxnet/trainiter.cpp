#include "trainiter.h"
#include "imageutils.h"

#include <Eigen/Dense>

#include <algorithm>

TrainIter::TrainIter(const mxnet::cpp::Context& ctx,
                     ImageDb* image_db,
                     const Params& params)
    : image_db_(image_db),
      batch_size_(params.rcnn_batch_size),
      short_side_len_(params.img_short_side),
      long_side_len_(params.img_long_side),
      one_image_size_(3 * short_side_len_ * long_side_len_),
      num_anchors_(static_cast<uint32_t>(params.rpn_anchor_scales.size() *
                                         params.rpn_anchor_ratios.size())),
      batch_gt_boxes_count_(params.rcnn_batch_gt_boxes),
      batch_indices_(batch_size_),
      anchor_generator_(params),
      anchor_sampler_(params),
      im_data_(
          mxnet::cpp::Shape(batch_size_, 3, short_side_len_, long_side_len_),
          ctx,
          false),
      im_info_data_(mxnet::cpp::Shape(batch_size_, 3), ctx, false) {
  assert(image_db_ != nullptr);
  size_ = image_db->GetImagesCount();
  Reset();
}

void TrainIter::Reset() {
  cur_ = 0;
  data_indices_.resize(size_);
  std::iota(data_indices_.begin(), data_indices_.end(), 0);
  std::shuffle(data_indices_.begin(), data_indices_.end(), random_engine_);
}

bool TrainIter::Next(uint32_t feat_height, uint32_t feat_width) {
  if (cur_ + batch_size_ <= size_) {
    auto s = data_indices_.begin() + cur_;
    auto e = s + batch_size_;
    std::copy(s, e, batch_indices_.begin());

    FillData();
    FillLabels(feat_height, feat_width);
    mxnet::cpp::NDArray::WaitAll();

    cur_ += batch_size_;
    return true;
  } else {
    return false;
  }
}

mxnet::cpp::NDArray TrainIter::GetImData() {
  return im_data_;
}

mxnet::cpp::NDArray TrainIter::GetImInfoData() {
  return im_info_data_;
}

mxnet::cpp::NDArray TrainIter::GetGtBoxesData() {
  return gt_boxes_data_;
}

mxnet::cpp::NDArray TrainIter::GetLabel() {
  return label_;
}

mxnet::cpp::NDArray TrainIter::GetBBoxTraget() {
  return bbox_target_;
}

mxnet::cpp::NDArray TrainIter::GetBBoxWeight() {
  return bbox_weight_;
}

void TrainIter::FillData() {
  raw_gt_boxes_data_.clear();
  raw_im_data_.clear();
  raw_im_info_data_.clear();

  auto ii = raw_im_data_.begin();
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
    raw_im_data_.insert(ii, array.begin(), array.end());
    std::advance(ii, one_image_size_);

    // Fill info
    *if_i++ = image_desc.height;
    *if_i++ = image_desc.width;
    *if_i++ = image_desc.scale;
    // Pad is not required

    // Fill boxes
    if (image_desc.boxes.size() > batch_gt_boxes_count_)
      image_desc.boxes.resize(batch_gt_boxes_count_);
    auto ic = image_desc.classes.begin();
    for (const auto& b : image_desc.boxes) {
      // sanitize box
      auto x1 = std::max(0.f, b.x);
      auto y1 = std::max(0.f, b.y);
      auto x2 = std::min(image_desc.width - 1, x1 + std::max(0.f, b.width - 1));
      auto y2 =
          std::min(image_desc.height - 1, y1 + std::max(0.f, b.height - 1));
      *b_i++ = x1 * image_desc.scale;
      *b_i++ = y1 * image_desc.scale;
      *b_i++ = x2 * image_desc.scale;
      *b_i++ = y2 * image_desc.scale;
      *b_i++ = *(ic++);  // class index
      gt_boxes_pad_markers.push_back(
          {raw_gt_boxes_data_.size(), image_desc.boxes.size()});
    }
  }
  im_data_.SyncCopyFromCPU(raw_im_data_.data(), raw_im_data_.size());

  im_info_data_.SyncCopyFromCPU(raw_im_info_data_.data(),
                                raw_im_info_data_.size());

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

  gt_boxes_data_ = mxnet::cpp::NDArray(
      mxnet::cpp::Shape(static_cast<mx_uint>(raw_gt_boxes_data_.size() / 5), 5),
      im_data_.GetContext(), false);
  gt_boxes_data_.SyncCopyFromCPU(raw_gt_boxes_data_.data(),
                                 raw_gt_boxes_data_.size());
  gt_boxes_data_ = gt_boxes_data_.Reshape(
      mxnet::cpp::Shape(batch_size_, batch_gt_boxes_count_, 5));
}

void TrainIter::FillLabels(uint32_t feat_height, uint32_t feat_width) {
  // all stacked image share same anchors
  auto anchors = anchor_generator_.Generate(feat_width, feat_height);

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
  auto all_boxes = Eigen::Map<Eigen::MatrixXf>(
      raw_gt_boxes_data_.data(),
      static_cast<mx_uint>(raw_gt_boxes_data_.size() / 5), 5);
  for (uint32_t i = 0, box_index = 0; i < batch_size_; ++i) {
    auto im_width = raw_im_info_data_[i * 4 + 1];
    auto im_height = raw_im_info_data_[i * 4];
    auto boxes = all_boxes.block(box_index, 0, batch_gt_boxes_count_, 4);
    box_index += batch_gt_boxes_count_;

    Eigen::MatrixXf b_label, b_bbox_target, b_bbox_weight;
    std::tie(b_label, b_bbox_target, b_bbox_weight) =
        anchor_sampler_.Assign(anchors, boxes, im_width, im_height);

    label_map.block(i * anchors.rows(), 0, anchors.rows(), 1) = b_label;

    bbox_target_map.block(i * anchors.rows(), 0, anchors.rows(), 4) =
        b_bbox_target;
    bbox_weight_map.block(i * anchors.rows(), 0, anchors.rows(), 4) =
        b_bbox_weight;
  }

  // copy batch to GPU
  label_ = mxnet::cpp::NDArray(
      mxnet::cpp::Shape(static_cast<mx_uint>(label_map.rows()),
                        static_cast<mx_uint>(label_map.cols())),
      im_data_.GetContext(), false);
  label_.SyncCopyFromCPU(raw_label_.data(), raw_label_.size());

  bbox_target_ = mxnet::cpp::NDArray(
      mxnet::cpp::Shape(static_cast<mx_uint>(bbox_target_map.rows()),
                        static_cast<mx_uint>(bbox_target_map.cols())),
      im_data_.GetContext(), false);
  bbox_target_.SyncCopyFromCPU(raw_bbox_target_.data(),
                               raw_bbox_target_.size());

  bbox_weight_ = mxnet::cpp::NDArray(
      mxnet::cpp::Shape(static_cast<mx_uint>(bbox_weight_map.rows()),
                        static_cast<mx_uint>(bbox_weight_map.cols())),
      im_data_.GetContext(), false);
  bbox_weight_.SyncCopyFromCPU(raw_bbox_weight_.data(),
                               raw_bbox_weight_.size());
  //  reshape
  label_ = label_.Reshape(mxnet::cpp::Shape(
      batch_size_, 1, num_anchors_ * feat_height, feat_width));
  bbox_target_ = bbox_target_.Reshape(mxnet::cpp::Shape(
      batch_size_, 4 * num_anchors_, feat_height, feat_width));
  bbox_weight_ = bbox_weight_.Reshape(mxnet::cpp::Shape(
      batch_size_, 4 * num_anchors_, feat_height, feat_width));
}
