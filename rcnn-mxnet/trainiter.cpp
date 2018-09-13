#include "trainiter.h"
#include "imageutils.h"

#include <Eigen/Dense>

#include <algorithm>

TrainIter::TrainIter(const mxnet::cpp::Context& ctx,
                     ImageDb* image_db,
                     uint32_t batch_size,
                     const Params& params)
    : image_db_(image_db),
      short_side_len_(params.img_short_side),
      long_side_len_(params.img_long_side),
      one_image_size_(3 * short_side_len_ * long_side_len_),
      batch_indices_(batch_size),
      anchor_generator_(params),
      anchor_sampler_(params),
      im_data_(
          mxnet::cpp::Shape(batch_size, 3, short_side_len_, long_side_len_),
          ctx,
          false),
      raw_im_data_(batch_size * 3 * short_side_len_ * long_side_len_),
      im_info_data_(mxnet::cpp::Shape(batch_size, 4), ctx, false),
      raw_im_info_data_(batch_size * 4) {
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

bool TrainIter::Next() {
  if (cur_ + batch_size_ <= size_) {
    auto s = data_indices_.begin() + cur_;
    auto e = s + batch_size_;
    std::copy(s, e, batch_indices_.begin());

    FillData();
    FillLabels();
    mxnet::cpp::NDArray::WaitAll();

    cur_ += batch_size_;
    return true;
  } else {
    return false;
  }
}

void TrainIter::FillData() {
  raw_gt_boxes_data_.clear();
  raw_im_data_.clear();
  raw_im_info_data_.clear();

  auto ii = raw_im_data_.begin();
  auto if_i = raw_im_info_data_.begin();
  auto b_i = std::back_insert_iterator<std::vector<float>>(raw_gt_boxes_data_);
  for (auto index : batch_indices_) {
    auto image_desc =
        image_db_->GetImage(index, short_side_len_, long_side_len_);
    // Fill image
    auto array = CVToMxnetFormat(image_desc.image);
    raw_im_data_.insert(ii, array.begin(), array.end());
    std::advance(ii, one_image_size_);
    // Fill info
    *if_i++ = image_desc.width;
    *if_i++ = image_desc.height;
    *if_i++ = image_desc.scale;
    *if_i++ = static_cast<float>(image_desc.boxes.size());
    // Fill boxes
    for (const auto& b : image_desc.boxes) {
      *b_i++ = b.x;
      *b_i++ = b.y;
      *b_i++ = b.width;
      *b_i++ = b.height;
    }
  }
  im_data_.SyncCopyFromCPU(raw_im_data_.data(), raw_im_data_.size());
  im_info_data_.SyncCopyFromCPU(raw_im_info_data_.data(),
                                raw_im_info_data_.size());
  // TODO - dim should be 5
  gt_boxes_data_ = mxnet::cpp::NDArray(
      mxnet::cpp::Shape(static_cast<mx_uint>(raw_gt_boxes_data_.size() / 4), 4),
      im_data_.GetContext(), false);
  gt_boxes_data_.SyncCopyFromCPU(raw_gt_boxes_data_.data(),
                                 raw_gt_boxes_data_.size());
}

void TrainIter::FillLabels() {
  // all stacked image share same anchors
  std::map<std::string, std::vector<mx_uint>> arg_shapes;
  arg_shapes.insert({std::string("data"), im_data_.GetShape()});
  std::vector<std::vector<mx_uint>> in_shape;
  std::vector<std::vector<mx_uint>> aux_shape;
  std::vector<std::vector<mx_uint>> out_shape;
  feat_sym_.InferShape(arg_shapes, &in_shape, &aux_shape, &out_shape);
  auto feat_height = out_shape[0][0];
  auto feat_width = out_shape[0][1];
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
      static_cast<mx_uint>(raw_gt_boxes_data_.size() / 4), 4);
  for (uint32_t i = 0, box_index = 0; i < batch_size_; ++i) {
    auto im_width = raw_im_info_data_[i * 4];
    auto im_height = raw_im_info_data_[i * 4 + 1];
    uint32_t boxes_count = static_cast<uint32_t>(raw_im_info_data_[i * 4 + 3]);
    auto boxes = all_boxes.block(box_index, 0, boxes_count, 4);
    box_index += boxes_count;

    Eigen::MatrixXf b_label, b_bbox_target, b_bbox_weight;
    std::tie(b_label, b_bbox_target, b_bbox_weight) =
        anchor_sampler_.Assign(anchors, boxes, im_width, im_height);

    label_map.block(i * anchors.rows(), 0, anchors.rows(), 1) = b_label;

    bbox_target_map.block(i * anchors.rows() * 4, 0, anchors.rows() * 4, 4) =
        b_bbox_target;
    bbox_weight_map.block(i * anchors.rows() * 4, 0, anchors.rows() * 4, 4) =
        b_bbox_weight;
  }

  // copy batch to GPU
  label_ = mxnet::cpp::NDArray(
      mxnet::cpp::Shape(static_cast<mx_uint>(raw_label_.size())),
      im_data_.GetContext(), false);
  label_.SyncCopyFromCPU(raw_label_.data(), raw_label_.size());

  bbox_target_ = mxnet::cpp::NDArray(
      mxnet::cpp::Shape(static_cast<mx_uint>(raw_bbox_target_.size())),
      im_data_.GetContext(), false);
  bbox_target_.SyncCopyFromCPU(raw_bbox_target_.data(),
                               raw_bbox_target_.size());

  bbox_weight_ = mxnet::cpp::NDArray(
      mxnet::cpp::Shape(static_cast<mx_uint>(raw_bbox_weight_.size())),
      im_data_.GetContext(), false);
  bbox_weight_.SyncCopyFromCPU(raw_bbox_weight_.data(),
                               raw_bbox_weight_.size());
}
