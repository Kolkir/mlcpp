#include "rcnn.h"
#include "resnet.h"

#include <nnvm/tuple.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

mxnet::cpp::Symbol GetRCNNSymbol(const Params& params) {
  using namespace mxnet::cpp;
  float num_anchors =
      params.rpn_anchor_scales.size() * params.rpn_anchor_ratios.size();

  auto data = Symbol::Variable("data");
  auto rpn_label = Symbol::Variable("label");
  auto im_info = Symbol::Variable("im_info");
  auto gt_boxes = Symbol::Variable("gt_boxes");
  auto rpn_bbox_target = Symbol::Variable("bbox_target");
  auto rpn_bbox_weight = Symbol::Variable("bbox_weight");

  std::vector<uint32_t> units{3, 4, 6, 3};
  std::vector<uint32_t> filter_list{256, 512, 1024, 2048};

  // shared convolutional layers
  Symbol conv_feat = GetResnetHeadSymbol(data, units, filter_list);

  // rpn feature
  auto rpn_conv = Operator("Convolution")
                      .SetParam("kernel", Shape(3, 3))
                      .SetParam("num_filter", 512)
                      .SetParam("pad", Shape(1, 1))
                      .SetInput("data", conv_feat)
                      .CreateSymbol("rpn_conv_3x3");
  auto rpn_relu = Operator("Activation")
                      .SetParam("act_type", "relu")
                      .SetInput("data", rpn_conv)
                      .CreateSymbol("rpn_relu");

  // rpn classification
  auto rpn_cls_score = Operator("Convolution")
                           .SetParam("kernel", Shape(1, 1))
                           .SetParam("num_filter", 2 * num_anchors)
                           .SetParam("pad", Shape(0, 0))
                           .SetInput("data", rpn_relu)
                           .CreateSymbol("rpn_cls_score");

  auto rpn_cls_score_reshape =
      Operator("Reshape")
          .SetParam("shape", Shape(0, 2, static_cast<index_t>(-1), 0))
          .SetInput("data", rpn_cls_score)
          .CreateSymbol("rpn_cls_score_reshape");

  auto rpn_cls_prob = Operator("SoftmaxOutput")
                          .SetParam("ignore_label", -1)
                          .SetParam("multi_output", true)
                          .SetParam("use_ignore", true)
                          .SetParam("normalization", "valid")
                          .SetInput("data", rpn_cls_score_reshape)
                          .SetInput("label", rpn_label)
                          .CreateSymbol("rpn_cls_prob");

  auto rpn_cls_act = Operator("softmax")
                         .SetParam("axis", 1)
                         .SetInput("data", rpn_cls_score_reshape)
                         .CreateSymbol("rpn_cls_act");

  auto rpn_cls_act_reshape =
      Operator("Reshape")
          .SetParam("shape", Shape(0, static_cast<index_t>(2 * num_anchors),
                                   static_cast<index_t>(-1), 0))
          .SetInput("data", rpn_cls_act)
          .CreateSymbol("rpn_cls_act_reshape");

  // rpn bbox regression
  auto rpn_bbox_pred = Operator("Convolution")
                           .SetParam("kernel", Shape(1, 1))
                           .SetParam("num_filter", 4 * num_anchors)
                           .SetParam("pad", Shape(0, 0))
                           .SetInput("data", rpn_relu)
                           .CreateSymbol("rpn_bbox_pred");

  auto rpn_bbox_loss_ =
      rpn_bbox_weight * Operator("smooth_l1")
                            .SetParam("scalar", 3)
                            .SetParam("num_filter", 4 * num_anchors)
                            .SetParam("pad", Shape(0, 0))
                            .SetInput("data", (rpn_bbox_pred - rpn_bbox_target))
                            .CreateSymbol("rpn_bbox_loss_");

  // You can also use normal function for creating symbols
  auto rpn_bbox_loss =
      MakeLoss("rpn_bbox_loss", rpn_bbox_loss_, 1 / params.rpn_batch_rois);

  // rpn proposal
  auto rois = Operator("_contrib_MultiProposal")
                  .SetParam("feature_stride", params.rpn_feat_stride)
                  .SetParam("scales",
                            nnvm::Tuple<float>(params.rpn_anchor_scales.begin(),
                                               params.rpn_anchor_scales.end()))
                  .SetParam("ratios",
                            nnvm::Tuple<float>(params.rpn_anchor_ratios.begin(),
                                               params.rpn_anchor_ratios.end()))
                  .SetParam("rpn_pre_nms_top_n", params.rpn_pre_nms_topk)
                  .SetParam("rpn_post_nms_top_n", params.rpn_post_nms_topk)
                  .SetParam("threshold", params.rpn_nms_thresh)
                  .SetParam("rpn_min_size", params.rpn_min_size)
                  .SetInput("cls_prob", rpn_cls_act_reshape)
                  .SetInput("bbox_pred", rpn_bbox_pred)
                  .SetInput("im_info", im_info)
                  .CreateSymbol("rois");

  // rcnn roi proposal target
  auto group = Operator("proposal_target")
                   .SetParam("num_classes", params.rcnn_num_classes)
                   .SetParam("batch_images", params.rcnn_batch_size)
                   .SetParam("batch_rois", params.rcnn_batch_rois)
                   .SetParam("fg_fraction", params.rcnn_fg_fraction)
                   .SetParam("fg_overlap", params.rcnn_fg_overlap)
                   .SetParam("box_stds",
                             nnvm::Tuple<float>(params.rcnn_bbox_stds.begin(),
                                                params.rcnn_bbox_stds.end()))
                   .SetInput("rois", rois)
                   .SetInput("gt_boxes", gt_boxes)
                   .CreateSymbol();

  rois = group[0];
  auto label = group[1];
  auto bbox_target = group[2];
  auto bbox_weight = group[3];

  // rcnn roi pool
  auto roi_pool =
      Operator("ROIPooling")
          .SetParam("pooled_size",
                    Shape(static_cast<index_t>(params.rcnn_pooled_size[0]),
                          static_cast<index_t>(params.rcnn_pooled_size[1])))
          .SetParam("spatial_scale", 1.0f / params.rpn_feat_stride)
          .SetInput("rois", rois)
          .SetInput("data", conv_feat)
          .CreateSymbol("roi_pool");

  // rcnn top feature
  mxnet::cpp::Symbol top_feat = GetResnetTopSymbol(data, units, filter_list);

  // rcnn classification
  auto cls_score = Operator("FullyConnected")
                       .SetParam("num_hidden", params.rcnn_num_classes)
                       .SetInput("data", top_feat)
                       .CreateSymbol("cls_score");

  auto cls_prob =
      SoftmaxOutput("cls_prob", cls_score, label, 1, -1, false, false, false,
                    SoftmaxOutputNormalization::kBatch);

  // rcnn bbox regression
  auto bbox_pred = Operator("FullyConnected")
                       .SetParam("num_hidden", params.rcnn_num_classes * 4)
                       .SetInput("data", top_feat)
                       .CreateSymbol("bbox_pred");

  auto bbox_loss_ =
      bbox_weight * Operator("smooth_l1")
                        .SetParam("scalar", 1.0f)
                        .SetInput("data", (bbox_pred - bbox_target))
                        .CreateSymbol("bbox_loss_");

  auto bbox_loss =
      MakeLoss("bbox_loss", bbox_loss_, 1.0f / params.rcnn_batch_rois);

  // reshape output
  label = Reshape("label_reshape", label,
                  Shape(static_cast<index_t>(params.rcnn_batch_size),
                        static_cast<index_t>(-1)));
  cls_prob = Reshape("cls_prob_reshape", cls_prob,
                     Shape(static_cast<index_t>(params.rcnn_batch_size),
                           static_cast<index_t>(-1),
                           static_cast<index_t>(params.rcnn_num_classes)));
  bbox_loss = Reshape("bbox_loss_reshape", bbox_loss,
                      Shape(static_cast<index_t>(params.rcnn_batch_size),
                            static_cast<index_t>(-1),
                            static_cast<index_t>(4 * params.rcnn_num_classes)));
  // group output

  group = Symbol::Group(
      {rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, BlockGrad(label)});
  return group;
}
