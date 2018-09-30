#include "rcnn.h"
#include "resnet.h"

#include <nnvm/tuple.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

static const mxnet::cpp::index_t undefined =
    static_cast<mxnet::cpp::index_t>(-1);

mxnet::cpp::Symbol GetRCNNSymbol(const Params& params, bool train) {
  using namespace mxnet::cpp;
  float num_anchors =
      params.rpn_anchor_scales.size() * params.rpn_anchor_ratios.size();

  auto data = Symbol::Variable("data");
  auto im_info = Symbol::Variable("im_info");
  auto gt_boxes = Symbol::Variable("gt_boxes");
  auto rpn_label = Symbol::Variable("label");
  auto rpn_bbox_target = Symbol::Variable("bbox_target");
  auto rpn_bbox_weight = Symbol::Variable("bbox_weight");

  std::vector<Symbol> out_group;

  // resnet 101
  std::vector<uint32_t> units{3, 4, 23, 3};
  std::vector<uint32_t> filter_list{256, 512, 1024, 2048};

  // shared convolutional layers
  Symbol conv_feat = GetResnetHeadSymbol(data, units, filter_list);

  // rpn feature
  auto rpn_conv = Operator("Convolution")
                      .SetParam("kernel", Shape(3, 3))
                      .SetParam("num_filter", 512)
                      .SetParam("pad", Shape(1, 1))
                      .SetInput("weight", Symbol("rpn_conv_3x3_weight"))
                      .SetInput("bias", Symbol("rpn_conv_3x3_bias"))
                      .SetInput("data", conv_feat)
                      .CreateSymbol("rpn_conv_3x3");

  auto rpn_relu = Operator("Activation")
                      .SetParam("act_type", "relu")
                      .SetInput("data", rpn_conv)
                      .CreateSymbol("rpn_relu");

  // rpn classification
  auto rpn_cls_score =
      Operator("Convolution")
          .SetParam("kernel", Shape(1, 1))
          .SetParam("num_filter", static_cast<uint32_t>(2 * num_anchors))
          .SetParam("pad", Shape(0, 0))
          .SetInput("weight", Symbol("rpn_cls_score_weight"))
          .SetInput("bias", Symbol("rpn_cls_score_bias"))
          .SetInput("data", rpn_relu)
          .CreateSymbol("rpn_cls_score");

  auto rpn_cls_score_reshape = Operator("Reshape")
                                   .SetParam("shape", Shape(0, 2, undefined, 0))
                                   .SetInput("data", rpn_cls_score)
                                   .CreateSymbol("rpn_cls_score_reshape");
  if (train) {
    auto rpn_cls_prob = Operator("SoftmaxOutput")
                            .SetParam("ignore_label", -1)
                            .SetParam("multi_output", true)
                            .SetParam("use_ignore", true)
                            .SetParam("normalization", "valid")
                            .SetInput("data", rpn_cls_score_reshape)
                            .SetInput("label", rpn_label)
                            .CreateSymbol("rpn_cls_prob");
    out_group.push_back(rpn_cls_prob);
  }

  auto rpn_cls_act = Operator("softmax")
                         .SetParam("axis", 1)
                         .SetInput("data", rpn_cls_score_reshape)
                         .CreateSymbol("rpn_cls_act");

  auto rpn_cls_act_reshape =
      Operator("Reshape")
          .SetParam("shape", Shape(0, static_cast<index_t>(2 * num_anchors),
                                   undefined, 0))
          .SetInput("data", rpn_cls_act)
          .CreateSymbol("rpn_cls_act_reshape");

  // rpn bbox regression
  auto rpn_bbox_pred =
      Operator("Convolution")
          .SetParam("kernel", Shape(1, 1))
          .SetParam("num_filter", static_cast<uint32_t>(4 * num_anchors))
          .SetParam("pad", Shape(0, 0))
          .SetInput("data", rpn_relu)
          .SetInput("weight", Symbol("rpn_bbox_pred_weight"))
          .SetInput("bias", Symbol("rpn_bbox_pred_bias"))
          .CreateSymbol("rpn_bbox_pred");

  if (train) {
    Symbol rpn_bbox_diff = rpn_bbox_pred - rpn_bbox_target;
    auto rpn_bbox_loss_ =
        rpn_bbox_weight * smooth_l1("rpn_bbox_loss_", rpn_bbox_diff, 3.f);

    auto rpn_bbox_loss =
        MakeLoss("rpn_bbox_loss", rpn_bbox_loss_, 1.f / params.rpn_batch_rois);
    out_group.push_back((rpn_bbox_loss));
  }

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

  Symbol label;
  Symbol bbox_target;
  Symbol bbox_weight;
  if (train) {
    // rcnn roi proposal target
    auto group = Operator("_contrib_proposal_target")
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
    label = group[1];
    bbox_target = group[2];
    bbox_weight = group[3];
  }

  // rcnn roi pool
  auto roi_pool =
      Operator("ROIPooling")
          .SetParam("pooled_size",
                    Shape(static_cast<index_t>(params.rcnn_pooled_size[0]),
                          static_cast<index_t>(params.rcnn_pooled_size[1])))
          .SetParam("spatial_scale", 1.0f / params.rcnn_feat_stride)
          .SetInput("rois", rois)
          .SetInput("data", conv_feat)
          .CreateSymbol("roi_pool");

  // rcnn top feature
  mxnet::cpp::Symbol top_feat =
      GetResnetTopSymbol(roi_pool, units, filter_list);

  // rcnn classification
  auto cls_score = Operator("FullyConnected")
                       .SetParam("num_hidden", params.rcnn_num_classes)
                       .SetParam("no_bias", false)
                       .SetParam("flatten", true)
                       .SetInput("weight", Symbol("cls_score_weight"))
                       .SetInput("bias", Symbol("cls_score_bias"))
                       .SetInput("data", Flatten(top_feat))
                       .CreateSymbol("cls_score");

  Symbol cls_prob;
  if (train) {
    cls_prob = Operator("SoftmaxOutput")
                   .SetParam("normalization", "batch")
                   .SetInput("label", label)
                   .SetInput("data", cls_score)
                   .CreateSymbol("cls_prob");
  } else {
    cls_prob = softmax("cls_prob", cls_score);
  }

  // rcnn bbox regression
  auto bbox_pred = Operator("FullyConnected")
                       .SetParam("num_hidden", params.rcnn_num_classes * 4)
                       .SetParam("no_bias", false)
                       .SetParam("flatten", true)
                       .SetInput("weight", Symbol("bbox_pred_weight"))
                       .SetInput("bias", Symbol("bbox_pred_bias"))
                       .SetInput("data", top_feat)
                       .CreateSymbol("bbox_pred");

  Symbol bbox_loss;
  if (train) {
    auto bbox_loss_ =
        bbox_weight * Operator("smooth_l1")
                          .SetParam("scalar", 1.0f)
                          .SetInput("data", (bbox_pred - bbox_target))
                          .CreateSymbol("bbox_loss_");
    bbox_loss =
        MakeLoss("bbox_loss", bbox_loss_, 1.0f / params.rcnn_batch_rois);

    bbox_loss =
        Reshape("bbox_loss_reshape", bbox_loss,
                Shape(static_cast<index_t>(params.rcnn_batch_size), undefined,
                      static_cast<index_t>(4 * params.rcnn_num_classes)));

    // reshape output
    label =
        Reshape("label_reshape", label,
                Shape(static_cast<index_t>(params.rcnn_batch_size), undefined));
  }

  // reshape output
  cls_prob =
      Reshape("cls_prob_reshape", cls_prob,
              Shape(static_cast<index_t>(params.rcnn_batch_size), undefined,
                    static_cast<index_t>(params.rcnn_num_classes)));

  if (train) {
    out_group.push_back(cls_prob);
    out_group.push_back(bbox_loss);
    out_group.push_back(BlockGrad(label));
  } else {
    out_group.push_back(rois);
    out_group.push_back(cls_prob);

    bbox_pred =
        Reshape("bbox_pred_reshape", bbox_pred,
                Shape(static_cast<index_t>(params.rcnn_batch_size), undefined,
                      static_cast<index_t>(4 * params.rcnn_num_classes)));

    out_group.push_back(bbox_pred);
  }

  // group output
  auto res_group = Symbol::Group(out_group);
  return res_group;
}

void InitiaizeRCNN(std::map<std::string, mxnet::cpp::NDArray>& args_map) {
  mxnet::cpp::Normal normal(0, 0.01f);
  mxnet::cpp::Zero zero;
  normal("rpn_conv_3x3_weight", &args_map["rpn_conv_3x3_weight"]);
  args_map["rpn_conv_3x3_weight"].WaitAll();
  zero("rpn_conv_3x3_bias", &args_map["rpn_conv_3x3_bias"]);
  args_map["rpn_conv_3x3_bias"].WaitAll();

  normal("rpn_cls_score_weight", &args_map["rpn_cls_score_weight"]);
  args_map["rpn_cls_score_weight"].WaitAll();
  zero("rpn_cls_score_bias", &args_map["rpn_cls_score_bias"]);
  args_map["rpn_cls_score_bias"].WaitAll();

  normal("rpn_bbox_pred_weight", &args_map["rpn_bbox_pred_weight"]);
  args_map["rpn_bbox_pred_weight"].WaitAll();
  zero("rpn_bbox_pred_bias", &args_map["rpn_bbox_pred_bias"]);
  args_map["rpn_bbox_pred_bias"].WaitAll();

  normal("cls_score_weight", &args_map["cls_score_weight"]);
  args_map["cls_score_weight"].WaitAll();
  zero("cls_score_bias", &args_map["cls_score_bias"]);
  args_map["cls_score_bias"].WaitAll();

  normal("bbox_pred_weight", &args_map["bbox_pred_weight"]);
  args_map["bbox_pred_weight"].WaitAll();
  zero("bbox_pred_bias", &args_map["rpn_conv_3x3_bias"]);
  args_map["rpn_conv_3x3_bias"].WaitAll();
}
