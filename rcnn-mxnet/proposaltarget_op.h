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

}  // namespace op
}  // namespace mxnet
#endif  // PROPOSALTARGET_H
