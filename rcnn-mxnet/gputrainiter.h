#ifndef GPUTRAINITER_H
#define GPUTRAINITER_H

#include <mxnet-cpp/MxNetCpp.h>

#include <atomic>
#include <condition_variable>
#include <future>
#include <thread>

#include "trainiter.h"

class GpuTrainIter {
 public:
  GpuTrainIter(ImageDb* image_db,
               const Params& params,
               uint32_t feat_height,
               uint32_t feat_width);
  GpuTrainIter(const GpuTrainIter&) = delete;
  GpuTrainIter& operator=(const GpuTrainIter&) = delete;

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

  void AllocateGpuCache(const mxnet::cpp::Context& ctx,
                        const std::vector<mx_uint>& im_shape,
                        const std::vector<mx_uint>& im_info_shape,
                        const std::vector<mx_uint>& gt_boxes_shape,
                        const std::vector<mx_uint>& label_shape,
                        const std::vector<mx_uint>& bbox_target_shape,
                        const std::vector<mx_uint>& bbox_weight_shape);

  void ScheduleLoadDataToGpu();

 private:
  TrainIter train_iter_;
  std::future<bool> load_result_;

  mxnet::cpp::NDArray gpu_im_arr;
  mxnet::cpp::NDArray gpu_im_info_arr;
  mxnet::cpp::NDArray gpu_gt_boxes_arr;
  mxnet::cpp::NDArray gpu_label_arr;
  mxnet::cpp::NDArray gpu_bbox_target_arr;
  mxnet::cpp::NDArray gpu_bbox_weight_arr;
};

#endif  // GPUTRAINITER_H
