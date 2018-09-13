#ifndef PROPOSALTARGET_H
#define PROPOSALTARGET_H

#include <mxnet/operator_util.h>
#include <nnvm/tuple.h>
#include "mshadow_op.h"
#include "operator_common.h"

namespace mxnet {
namespace op {
struct ProposalTargetParam : public dmlc::Parameter<ProposalTargetParam> {
  int num_classes{0};
  int batch_images{0};
  int batch_rois{0};
  float fg_fraction{0};
  float fg_overlap{0};
  nnvm::Tuple<float> box_stds;
  DMLC_DECLARE_PARAMETER(ProposalTargetParam) {
    DMLC_DECLARE_FIELD(num_classes)
        .set_default(81)
        .describe("Number of classes.");
    DMLC_DECLARE_FIELD(batch_images).set_default(1).describe("Batch images.");
    DMLC_DECLARE_FIELD(batch_rois).set_default(128).describe("Batch rois.");
    DMLC_DECLARE_FIELD(fg_fraction)
        .set_default(0.25)
        .describe("Foreground fraction.");
    DMLC_DECLARE_FIELD(fg_overlap)
        .set_default(0.5)
        .describe("Foreground overlap.");
    float tmp[] = {0.1f, 0.1f, 0.2f, 0.2f};
    DMLC_DECLARE_FIELD(box_stds)
        .set_default(nnvm::Tuple<float>(tmp, tmp + 4))
        .describe("Boxes stds.");
  }
};

template <typename xpu>
Operator* CreateOp(ProposalTargetParam param);

#if DMLC_USE_CXX11
class ProposalTargetProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs)
      override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape>* in_shape,
                  std::vector<TShape>* out_shape,
                  std::vector<TShape>* /*aux_shape*/) const override {
    CHECK_EQ(param_.batch_rois % param_.batch_images, 0);

    auto rpn_rois_shape = in_shape->at(0);
    auto gt_boxes_shape = in_shape->at(1);

    TShape output_rois_shape{param_.batch_rois, 5};
    TShape label_shape{param_.batch_rois};
    TShape bbox_target_shape{param_.batch_rois, param_.num_classes * 4};
    TShape bbox_weight_shape{param_.batch_rois, param_.num_classes * 4};

    SHAPE_ASSIGN_CHECK(*out_shape, 0, output_rois_shape);
    SHAPE_ASSIGN_CHECK(*out_shape, 1, label_shape);
    SHAPE_ASSIGN_CHECK(*out_shape, 2, bbox_target_shape);
    SHAPE_ASSIGN_CHECK(*out_shape, 3, bbox_weight_shape);

    return in_shape->at(0).ndim() != 0U && out_shape->at(0).Size() != 0U;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ProposalTargetProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override { return "_contrib_proposal_target"; }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape>& /*in_shape*/) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int>& /*out_grad*/,
      const std::vector<int>& /*in_data*/,
      const std::vector<int>& /*out_data*/) const override {
    return {};
  }

  int NumVisibleOutputs() const override { return 4; }

  int NumOutputs() const override { return 4; }

  std::vector<std::string> ListArguments() const override {
    return {"rois", "gt_boxes"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"rois_output", "label", "bbox_target", "bbox_weight"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  ProposalTargetParam param_;
};  // class ProposalProp

#endif  // DMLC_USE_CXX11

// template <typename Dtype>
// inline void ProposalTargetOpForwardImpl(
//    const mxnet::op::ProposalTargetParam& params,
//    const mshadow::Tensor<cpu, 2, Dtype>& all_rois_t,
//    const mshadow::Tensor<cpu, 2, Dtype>& all_gt_boxes_t,
//    const mshadow::Tensor<cpu, 2, Dtype>& b_rois_t,
//    const mshadow::Tensor<cpu, 1, Dtype>& b_labels_t,
//    const mshadow::Tensor<cpu, 2, Dtype>& b_bbox_targets_t,
//    const mshadow::Tensor<cpu, 2, Dtype>& b_bbox_weights_t) {}

// template <typename xpu>
// void ProposalTargetOpForward(const nnvm::NodeAttrs& attrs,
//                             const OpContext& ctx,
//                             const std::vector<TBlob>& inputs,
//                             const std::vector<OpReqType>& req,
//                             const std::vector<TBlob>& outputs) {
//  const ProposalTargetParam& params =
//      nnvm::get<ProposalTargetParam>(attrs.parsed);
//  CHECK_EQ(params.batch_images, inputs[1].shape_[0]);
//  using namespace mshadow;
//  mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
//  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
//    // define result shapes
//    auto rois_shape = Shape2(static_cast<index_t>(params.batch_rois), 5);
//    auto label_shape = Shape1(static_cast<index_t>(params.batch_rois));
//    auto bbox_target_shape =
//        Shape2(static_cast<index_t>(params.batch_rois),
//               static_cast<index_t>(params.num_classes * 4));
//    auto bbox_weight_shape =
//        Shape2(static_cast<index_t>(params.batch_rois),
//               static_cast<index_t>(params.num_classes * 4));

//    // allocate intermediate data
//    Tensor<xpu, 2, DType> rois =
//        inputs[0].get_with_shape<xpu, 2, DType>(rois_shape, s);
//    Tensor<xpu, 1, DType> labels =
//        inputs[0].get_with_shape<xpu, 1, DType>(label_shape, s);
//    Tensor<xpu, 2, DType> bbox_targets =
//        inputs[0].get_with_shape<xpu, 2, DType>(bbox_target_shape, s);
//    Tensor<xpu, 2, DType> bbox_weights =
//        inputs[0].get_with_shape<xpu, 2, DType>(bbox_weight_shape, s);

//    // get input tensors
//    // rois [n, 5] (batch_index, x1, y1, x2, y2)
//    Tensor<xpu, 2, DType> all_rois = inputs[0].get<xpu, 2, DType>(s);
//    // gt_boxes [n, 5] (x1, y1, x2, y2, cls)
//    Tensor<xpu, 2, DType> all_gt_boxes = inputs[1].get<xpu, 2, DType>(s);

//    // get destination tensors
//    Tensor<xpu, 2, DType> out0 =
//        outputs[0].get_with_shape<xpu, 2, DType>(rois_shape, s);
//    Tensor<xpu, 1, DType> out1 =
//        outputs[1].get_with_shape<xpu, 1, DType>(label_shape, s);
//    Tensor<xpu, 2, DType> out2 =
//        outputs[2].get_with_shape<xpu, 2, DType>(bbox_target_shape, s);
//    Tensor<xpu, 2, DType> out3 =
//        outputs[3].get_with_shape<xpu, 2, DType>(bbox_weight_shape, s);

//    // Main logic
//    CHECK_EQ(all_rois.CheckContiguous(), true);
//    CHECK_EQ(all_gt_boxes.CheckContiguous(), true);
//    CHECK_EQ(out0.CheckContiguous(), true);
//    CHECK_EQ(out1.CheckContiguous(), true);
//    CHECK_EQ(out2.CheckContiguous(), true);
//    CHECK_EQ(out3.CheckContiguous(), true);

////    ProposalTargetOpForwardImpl(params, all_rois, all_gt_boxes, out0, out1,
////                                out2, out3);

//    // copy results
//    ASSIGN_DISPATCH(out0, req[0], rois);
//    ASSIGN_DISPATCH(out1, req[1], labels);
//    ASSIGN_DISPATCH(out2, req[2], bbox_targets);
//    ASSIGN_DISPATCH(out3, req[3], bbox_weights);
//  });
//}

}  // namespace op
}  // namespace mxnet
#endif  // PROPOSALTARGET_H
