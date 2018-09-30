#include "trainiter.h"
#include "imageutils.h"

#include <Eigen/Dense>

#include <algorithm>

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
  StopCacheThread();
  cur_ = 0;
  data_indices_.resize(size_);
  std::iota(data_indices_.begin(), data_indices_.end(), 0);
  std::shuffle(data_indices_.begin(), data_indices_.end(), random_engine_);
  InitializeCache();
}

bool TrainIter::NextImpl(BatchData* data) {
  if (cur_ + batch_size_ <= size_) {
    auto s = data_indices_.begin() + cur_;
    auto e = s + batch_size_;
    std::copy(s, e, batch_indices_.begin());

    FillData(data);
    FillLabels(data);

    cur_ += batch_size_;
    return true;
  } else {
    return false;
  }
}

bool TrainIter::Next() {
  std::unique_lock<std::mutex> lock(available_data_guard_);
  available_data_cond_.wait(lock,
                            [&]() -> bool { return !available_data_.empty(); });
  current_data_ = available_data_.front();
  available_data_.pop();
  if (current_data_) {
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
  assert(current_data_);
  im_arr.SyncCopyFromCPU(current_data_->raw_im_data_.data(),
                         current_data_->raw_im_data_.size());
  im_arr.WaitAll();
  im_info_arr.SyncCopyFromCPU(current_data_->raw_im_info_data_.data(),
                              current_data_->raw_im_info_data_.size());
  im_info_arr.WaitAll();
  gt_boxes_arr.SyncCopyFromCPU(current_data_->raw_gt_boxes_data_.data(),
                               current_data_->raw_gt_boxes_data_.size());
  gt_boxes_arr.WaitAll();
  label_arr.SyncCopyFromCPU(current_data_->raw_label_.data(),
                            current_data_->raw_label_.size());
  label_arr.WaitAll();
  bbox_target_arr.SyncCopyFromCPU(current_data_->raw_bbox_target_.data(),
                                  current_data_->raw_bbox_target_.size());
  bbox_target_arr.WaitAll();
  bbox_weight_arr.SyncCopyFromCPU(current_data_->raw_bbox_weight_.data(),
                                  current_data_->raw_bbox_weight_.size());
  bbox_weight_arr.WaitAll();

  // return pointer to refill
  std::unique_lock<std::mutex> lock(free_data_guard_);
  free_data_.push(current_data_);
  free_data_cond_.notify_one();
}

void TrainIter::FillData(BatchData* data) {
  assert(data);
  data->raw_gt_boxes_data_.clear();
  data->raw_im_data_.clear();
  data->raw_im_info_data_.clear();

  auto if_i =
      std::back_insert_iterator<std::vector<float>>(data->raw_im_info_data_);
  auto b_i =
      std::back_insert_iterator<std::vector<float>>(data->raw_gt_boxes_data_);

  // gt_boxes padding markers
  std::vector<std::pair<size_t, size_t>> gt_boxes_pad_markers;

  for (auto index : batch_indices_) {
    // image is loaded with padding
    auto image_desc =
        image_db_->GetImage(index, short_side_len_, long_side_len_);
    // Fill image
    auto array = CVToMxnetFormat(image_desc.image);
    assert(array.size() <= one_image_size_);
    data->raw_im_data_.insert(data->raw_im_data_.end(), array.begin(),
                              array.end());

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
      auto x1 = std::max(0.f, b.x * image_desc.scale);
      auto y1 = std::max(0.f, b.y * image_desc.scale);
      auto x2 = std::min(image_desc.width - 1,
                         x1 + std::max(0.f, b.width * image_desc.scale - 1));
      auto y2 = std::min(image_desc.height - 1,
                         y1 + std::max(0.f, b.height * image_desc.scale - 1));
      *b_i++ = x1;
      *b_i++ = y1;
      *b_i++ = x2;
      *b_i++ = y2;

      auto class_index = *(ic++);
      *b_i++ = class_index;  // class index

      //--TEST
      //      cv::Point tl(static_cast<int>(x1), static_cast<int>(y1));
      //      cv::Point br(static_cast<int>(x2), static_cast<int>(y2));
      //      cv::Mat imgCopy = image_desc.image.clone();
      //      cv::rectangle(imgCopy, tl, br, cv::Scalar(1, 0, 0));
      //      cv::putText(imgCopy, std::to_string(class_index),
      //                  cv::Point(tl.x + 5, tl.y + 5),   // Coordinates
      //                  cv::FONT_HERSHEY_COMPLEX_SMALL,  // Font
      //                  1.0,                             // Scale. 2.0 = 2x
      //                  bigger cv::Scalar(100, 100, 255));      // BGR Color
      //      cv::imwrite("det.png", imgCopy);
      //--TEST
    }
    gt_boxes_pad_markers.push_back(
        {data->raw_gt_boxes_data_.size(), image_desc.boxes.size()});
  }

  // Add padding to gt_boxes
  size_t pad_pos = 0;
  for (auto& pad_marker : gt_boxes_pad_markers) {
    auto pad_size = batch_gt_boxes_count_ - pad_marker.second;
    if (pad_size != 0) {
      auto pos = data->raw_gt_boxes_data_.begin();
      std::advance(pos, pad_marker.first + pad_pos);
      data->raw_gt_boxes_data_.insert(pos, pad_size * 5, -1.f);
      pad_pos += pad_size * 5;
    }
  }
}

void TrainIter::FillLabels(BatchData* data) {
  assert(data);
  // all stacked image share same anchors
  auto anchors = anchor_generator_.Generate(feat_width_, feat_height_);

  // prepare data bindings
  data->raw_label_.resize(static_cast<size_t>(anchors.rows()) * batch_size_);
  Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      label_map(data->raw_label_.data(), anchors.rows() * batch_size_, 1);

  data->raw_bbox_target_.resize(
      static_cast<size_t>(anchors.rows() * 4 * batch_size_));
  Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      bbox_target_map(data->raw_bbox_target_.data(),
                      anchors.rows() * batch_size_, 4);

  data->raw_bbox_weight_.resize(
      static_cast<size_t>(anchors.rows() * 4 * batch_size_));
  Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      bbox_weight_map(data->raw_bbox_weight_.data(),
                      anchors.rows() * batch_size_, 4);

  // assign anchor according to their real size encoded in im_info
  auto all_boxes = Eigen::Map<Eigen::MatrixXf>(
      data->raw_gt_boxes_data_.data(),
      static_cast<mx_uint>(data->raw_gt_boxes_data_.size() / 5), 5);
  for (uint32_t i = 0, box_index = 0; i < batch_size_; ++i) {
    auto im_width = data->raw_im_info_data_[i * 3 + 1];
    auto im_height = data->raw_im_info_data_[i * 3];
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
}

void TrainIter::InitializeCache() {
  StopCacheThread();
  {
    current_data_ = nullptr;
    while (!available_data_.empty())
      available_data_.pop();

    while (!free_data_.empty())
      free_data_.pop();

    for (auto& d : data_cache_) {
      free_data_.push(&d);
    }
    free_data_cond_.notify_one();
  }
  StartCacheThread();
}

void TrainIter::StartCacheThread() {
  assert(!cache_thread_);
  stop_cache_flag_ = false;
  cache_thread_ =
      std::make_unique<std::thread>(std::bind(&TrainIter::CacheProc, this));
}

void TrainIter::StopCacheThread() {
  if (cache_thread_) {
    stop_cache_flag_ = true;
    cache_thread_->join();
    cache_thread_.reset();
  }
}

void TrainIter::CacheProc() {
  bool done{false};
  while (!done) {
    // thread cancellation
    if (stop_cache_flag_) {
      done = true;
      break;
    }
    // get free data slot to fill
    BatchData* data_to_fill{nullptr};
    {
      std::unique_lock<std::mutex> lock(free_data_guard_);
      free_data_cond_.wait(lock, [&]() { return !free_data_.empty(); });
      if (!free_data_.empty()) {
        data_to_fill = free_data_.front();
        free_data_.pop();
      }
    }
    // fill with data
    if (data_to_fill) {
      if (NextImpl(data_to_fill)) {
        std::unique_lock<std::mutex> lock(available_data_guard_);
        available_data_.push(data_to_fill);
        available_data_cond_.notify_one();
      } else {
        std::unique_lock<std::mutex> lock(available_data_guard_);
        available_data_.push(nullptr);
        available_data_cond_.notify_one();
        done = false;
        break;
      }
    }
  }
}
