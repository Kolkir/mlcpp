#ifndef TRAINITER_H
#define TRAINITER_H

#include "anchorgenerator.h"
#include "anchorsampler.h"
#include "imagedb.h"
#include "params.h"

#include <mxnet-cpp/MxNetCpp.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <random>
#include <thread>

struct BatchData {
  // train input
  std::vector<float> raw_im_data_;
  std::vector<float> raw_im_info_data_;
  std::vector<float> raw_gt_boxes_data_;

  // train labels
  std::vector<float> raw_label_;
  std::vector<float> raw_bbox_target_;
  std::vector<float> raw_bbox_weight_;
};

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
  bool NextImpl(BatchData* data);
  void FillData(BatchData* data);
  void FillLabels(BatchData* data);
  void InitializeCache();
  void StartCacheThread();
  void StopCacheThread();
  void CacheProc();

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

  const static size_t cache_size_{32};
  std::array<BatchData, cache_size_> data_cache_;
  std::queue<BatchData*> available_data_;
  std::queue<BatchData*> free_data_;
  BatchData* current_data_{nullptr};

  std::mutex free_data_guard_;
  std::mutex available_data_guard_;
  std::condition_variable available_data_cond_;
  std::condition_variable free_data_cond_;
  std::unique_ptr<std::thread> cache_thread_;
  std::atomic_bool stop_cache_flag_{false};
};

#endif  // TRAINITER_H
