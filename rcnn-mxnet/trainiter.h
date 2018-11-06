#ifndef TRAINITER_H
#define TRAINITER_H

#include "anchorgenerator.h"
#include "anchorsampler.h"
#include "imagedb.h"
#include "params.h"

#include <mxnet-cpp/MxNetCpp.h>

#include <array>
#include <random>

class TrainIter {
 public:
  TrainIter(ImageDb* image_db,
            const Params& params,
            uint32_t feat_height,
            uint32_t feat_width);
  TrainIter(const TrainIter&) = delete;
  TrainIter& operator=(const TrainIter&) = delete;

  uint32_t GetSize() const;
  uint32_t GetBatchCount() const;

  void Reset();
  bool Next();

  void GetData(mxnet::cpp::NDArray& im_arr,
               mxnet::cpp::NDArray& im_info_arr,
               mxnet::cpp::NDArray& gt_boxes_arr,
               mxnet::cpp::NDArray& label_arr,
               mxnet::cpp::NDArray& bbox_target_arr,
               mxnet::cpp::NDArray& bbox_weight_arr);

 private:
  void FillData();
  void FillLabels();
  void ReshapeLabels(unsigned int rows, unsigned int cols);
  void ReshapeTargets(unsigned int rows, unsigned int cols);
  void ReshapeWeights(unsigned int rows, unsigned int cols);

 private:
  ImageDb* image_db_{nullptr};
  uint32_t batch_size_{0};
  uint32_t size_{0};
  uint32_t cur_{0};

  uint32_t short_side_len_{0};
  uint32_t long_side_len_{0};
  uint32_t one_image_size_{0};
  uint32_t num_anchors_{0};
  uint32_t batch_gt_boxes_count_{0};
  uint32_t feat_height_;
  uint32_t feat_width_;

  size_t seed_ = 5675317;
  std::mt19937 random_engine_{seed_};
  std::vector<uint32_t> data_indices_;
  std::vector<uint32_t> batch_indices_{batch_size_};

  AnchorGenerator anchor_generator_;
  AnchorSampler anchor_sampler_;

  // train input
  std::vector<float> raw_im_data_;
  std::vector<float> raw_im_info_data_;
  std::vector<float> raw_gt_boxes_data_;

  // train labels
  std::vector<float> raw_label_;
  std::vector<float> raw_bbox_target_;
  std::vector<float> raw_bbox_weight_;
};

#endif  // TRAINITER_H
