#ifndef PROPOSALTARGET_H
#define PROPOSALTARGET_H

#include <mxnet/operator_util.h>
//#include <mxnet_op.h>

#include <mshadow_op.h>
#include <operator_common.h>

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
        .set_default(21)
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

inline bool ProposalTargetOpShape(const nnvm::NodeAttrs& attrs,
                                  std::vector<TShape>* in_attrs,
                                  std::vector<TShape>* out_attrs) {
  const ProposalTargetParam& param =
      nnvm::get<ProposalTargetParam>(attrs.parsed);
  CHECK_EQ(param.batch_rois % param.batch_images, 0);

  auto rpn_rois_shape = in_attrs->at(0);
  auto gt_boxes_shape = in_attrs->at(1);

  TShape output_rois_shape{param.batch_rois, 5};
  TShape label_shape{param.batch_rois};
  TShape bbox_target_shape{param.batch_rois, param.num_classes * 4};
  TShape bbox_weight_shape{param.batch_rois, param.num_classes * 4};

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, output_rois_shape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, label_shape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 2, bbox_target_shape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 3, bbox_weight_shape);

  return out_attrs->at(0).ndim() != 0U && out_attrs->at(0).Size() != 0U;
}

inline bool ProposalTargetOpType(const nnvm::NodeAttrs& attrs,
                                 std::vector<int>* in_attrs,
                                 std::vector<int>* out_attrs) {
  // TODO: implement
  //  CHECK_EQ(in_attrs->size(), 1U);
  //  CHECK_EQ(out_attrs->size(), 1U);

  //  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  //  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0) != -1;
}

template <typename xpu>
void ProposalTargetOpForward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  const ProposalTargetParam& param =
      nnvm::get<ProposalTargetParam>(attrs.parsed);
  CHECK_EQ(param.batch_images, inputs[1].shape_[0]);
  //  using namespace mshadow;
  //  MSHADOW_REAL_TYPE_SWITCH(
  //      outputs[0].type_flag_, DType,
  //      {
  //           Tensor<xpu, 2, DType> out =
  //                  outputs[0].get_with_shape<xpu, 2,
  //                  DType>(Shape2(shape[0], shape[2]), s);
  //              Tensor<xpu, 3, DType> rois =
  //                  inputs[0].get_with_shape<xpu, 3, DType>(shape.get<3>(),
  //                  s);
  //              CHECK(req[0] != kAddTo) << "AddTo is not supported";
  //              ASSIGN_DISPATCH(out, req[0], rois);
  //      });
}

template <int req>
struct proposal_target_backward {
  template <typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* in_grad) {
    KERNEL_ASSIGN(in_grad[i], req, 0);
  }
};

template <typename xpu>
void ProposalTargetOpBackward(const nnvm::NodeAttrs& /*attrs*/,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
  const TBlob& out_grad = inputs[0];
  const TBlob& in_grad = outputs[0];

  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(out_grad.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<proposal_target_backward<req_type>, xpu>::Launch(
          s, in_grad.Size(), in_grad.dptr<DType>());
    });
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // PROPOSALTARGET_H
