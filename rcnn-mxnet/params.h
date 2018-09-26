#ifndef PARAMS_H
#define PARAMS_H

#include <mxnet-cpp/MxNetCpp.h>

#include <stdint.h>
#include <string>
#include <vector>

class Params {
 public:
  uint32_t img_short_side = 480;
  uint32_t img_long_side = 640;
  float rpn_feat_stride = 16;
  std::vector<float> rpn_anchor_scales{8.f, 16.f, 32.f};
  std::vector<float> rpn_anchor_ratios{0.5f, 1.f, 2.f};
  int rpn_pre_nms_topk = 12000;
  int rpn_post_nms_topk = 2000;
  float rpn_nms_thresh = 0.7f;
  int rpn_min_size = 16;
  int rpn_batch_rois = 256;
  int rpn_allowed_border = 0;
  float rpn_fg_fraction = 0.5f;
  float rpn_fg_overlap = 0.7f;
  float rpn_bg_overlap = 0.3f;
  int rcnn_num_classes = 81;  // coco classes
  int rcnn_feat_stride = 16;
  std::vector<float> rcnn_pooled_size{14, 14};
  uint32_t rcnn_batch_size = 4;
  uint32_t rcnn_batch_gt_boxes = 100;
  int rcnn_batch_rois = 128;
  float rcnn_fg_fraction = 0.25f;
  float rcnn_fg_overlap = 0.5f;
  std::vector<float> rcnn_bbox_stds{0.1f, 0.1f, 0.2f, 0.2f};
  float rcnn_conf_thresh = 1e-3f;

  Params(bool is_eval = false) {
    if (is_eval) {
      rpn_pre_nms_topk = 6000;
      rpn_post_nms_topk = 300;
      rpn_nms_thresh = 0.3f;
      rcnn_batch_size = 1;
    }
  }
};

std::pair<std::map<std::string, mxnet::cpp::NDArray>,
          std::map<std::string, mxnet::cpp::NDArray>>
LoadNetParams(const mxnet::cpp::Context& ctx, const std::string& param_file);

void SaveNetParams(const std::string& param_file, mxnet::cpp::Executor* exe);

#endif  // PARAMS_H
