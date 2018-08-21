#ifndef TRAINITER_H
#define TRAINITER_H

#include "anchorgenerator.h"
#include "anchorsampler.h"
#include "imagedb.h"

#include <mxnet-cpp/MxNetCpp.h>

#include <random>

class TrainIter {
 public:
  TrainIter(const mxnet::cpp::Context& ctx,
            ImageDb* image_db,
            uint32_t batch_size);
  TrainIter(const TrainIter&) = delete;
  TrainIter& operator=(const TrainIter&) = delete;

  // DataIter interface
  void Reset();
  bool Next();
  mxnet::cpp::NDArray GetImData();
  mxnet::cpp::NDArray GetImInfoData();
  mxnet::cpp::NDArray GetGtBoxesData();

  mxnet::cpp::NDArray GetLabel();
  mxnet::cpp::NDArray GetBBoxTragetLabel();
  mxnet::cpp::NDArray GetBBoxWeightLabel();

 private:
  void FillData();
  void FillLabels();

 private:
  ImageDb* image_db_{nullptr};
  uint32_t batch_size_{0};
  uint32_t size_{0};
  uint32_t cur_{0};

  uint32_t short_side_len_{600};
  uint32_t long_side_len_{1000};
  uint32_t one_image_size_{3 * short_side_len_ * long_side_len_};

  size_t seed_ = 5675317;
  std::mt19937 random_engine_{seed_};
  std::vector<uint32_t> data_indices_;
  std::vector<uint32_t> batch_indices_{batch_size_};

  mxnet::cpp::Symbol feat_sym_;
  AnchorGenerator anchor_generator_;
  AnchorSampler anchor_sampler_;

  // train input
  mxnet::cpp::NDArray im_data_;
  std::vector<float> raw_im_data_;

  mxnet::cpp::NDArray im_info_data_;
  std::vector<float> raw_im_info_data_;

  mxnet::cpp::NDArray gt_boxes_data_;
  std::vector<float> raw_gt_boxes_data_;

  // train labels
  mxnet::cpp::NDArray label_;
  std::vector<float> raw_label_;

  mxnet::cpp::NDArray bbox_target_;
  std::vector<float> raw_bbox_target_;

  mxnet::cpp::NDArray bbox_weight_;
  std::vector<float> raw_bbox_weight_;
};

#endif  // TRAINITER_H
