#include "gputrainiter.h"

GpuTrainIter::GpuTrainIter(ImageDb* image_db,
                           const Params& params,
                           uint32_t feat_height,
                           uint32_t feat_width)
    : train_iter_(image_db, params, feat_height, feat_width) {}

uint32_t GpuTrainIter::GetSize() const {
  return train_iter_.GetSize();
}

uint32_t GpuTrainIter::GetBatchCount() const {
  return train_iter_.GetBatchCount();
}

void GpuTrainIter::Reset() {
  train_iter_.Reset();
  ScheduleLoadDataToGpu();
}

bool GpuTrainIter::Next() {
  auto data_available = load_result_.get();
  return data_available;
}

void GpuTrainIter::ScheduleLoadDataToGpu() {
  auto load_result = std::async(std::launch::async, [&]() {
    if (train_iter_.Next()) {
      // load data to GPU cache
      train_iter_.GetData(gpu_im_arr, gpu_im_info_arr, gpu_gt_boxes_arr,
                          gpu_label_arr, gpu_bbox_target_arr,
                          gpu_bbox_weight_arr);
      return true;
    } else {
      return false;
    }
  });
  load_result_ = std::move(load_result);
}

void GpuTrainIter::GetData(mxnet::cpp::NDArray& im_arr,
                           mxnet::cpp::NDArray& im_info_arr,
                           mxnet::cpp::NDArray& gt_boxes_arr,
                           mxnet::cpp::NDArray& label_arr,
                           mxnet::cpp::NDArray& bbox_target_arr,
                           mxnet::cpp::NDArray& bbox_weight_arr) {
  // Copy data from GPU to GPU
  gpu_im_arr.CopyTo(&im_arr);
  gpu_im_info_arr.CopyTo(&im_info_arr);
  gpu_gt_boxes_arr.CopyTo(&gt_boxes_arr);
  gpu_label_arr.CopyTo(&label_arr);
  gpu_bbox_target_arr.CopyTo(&bbox_target_arr);
  gpu_bbox_weight_arr.CopyTo(&bbox_weight_arr);
  mxnet::cpp::NDArray::WaitAll();

  auto* err = MXGetLastError();
  if (err && err[0] != 0) {
    std::cout << "MXNet unhandled error in GpuTrainIter::GetData : " << err
              << std::endl;
    exit(-1);
  }

  ScheduleLoadDataToGpu();
}

void GpuTrainIter::AllocateGpuCache(
    const mxnet::cpp::Context& ctx,
    const std::vector<mx_uint>& im_shape,
    const std::vector<mx_uint>& im_info_shape,
    const std::vector<mx_uint>& gt_boxes_shape,
    const std::vector<mx_uint>& label_shape,
    const std::vector<mx_uint>& bbox_target_shape,
    const std::vector<mx_uint>& bbox_weight_shape) {
  gpu_im_arr = mxnet::cpp::NDArray(im_shape, ctx, false);
  gpu_im_info_arr = mxnet::cpp::NDArray(im_info_shape, ctx, false);
  gpu_gt_boxes_arr = mxnet::cpp::NDArray(gt_boxes_shape, ctx, false);
  gpu_label_arr = mxnet::cpp::NDArray(label_shape, ctx, false);
  gpu_bbox_target_arr = mxnet::cpp::NDArray(bbox_target_shape, ctx, false);
  gpu_bbox_weight_arr = mxnet::cpp::NDArray(bbox_weight_shape, ctx, false);
}
