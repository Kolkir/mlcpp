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

inline bool ProposalTargetOpType(const nnvm::NodeAttrs& /*attrs*/,
                                 std::vector<int>* in_attrs,
                                 std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 4U);

  TYPE_ASSIGN_CHECK(*in_attrs, 0, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_attrs, 1, mshadow::kFloat32);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*out_attrs, 1, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*out_attrs, 2, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*out_attrs, 3, in_attrs->at(0));

  return out_attrs->at(0) != -1;
}

template <typename xpu>
void ProposalTargetOpForward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  const ProposalTargetParam& params =
      nnvm::get<ProposalTargetParam>(attrs.parsed);
  CHECK_EQ(params.batch_images, inputs[1].shape_[0]);
  using namespace mshadow;
  mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    // define result shapes
    auto rois_shape = Shape2(static_cast<index_t>(params.batch_rois), 5);
    auto label_shape = Shape1(static_cast<index_t>(params.batch_rois));
    auto bbox_target_shape =
        Shape2(static_cast<index_t>(params.batch_rois),
               static_cast<index_t>(params.num_classes * 4));
    auto bbox_weight_shape =
        Shape2(static_cast<index_t>(params.batch_rois),
               static_cast<index_t>(params.num_classes * 4));

    // allocate intermediate data
    Tensor<xpu, 2, DType> rois =
        inputs[0].get_with_shape<xpu, 2, DType>(rois_shape, s);
    Tensor<xpu, 1, DType> labels =
        inputs[0].get_with_shape<xpu, 1, DType>(label_shape, s);
    Tensor<xpu, 2, DType> bbox_targets =
        inputs[0].get_with_shape<xpu, 2, DType>(bbox_target_shape, s);
    Tensor<xpu, 2, DType> bbox_weights =
        inputs[0].get_with_shape<xpu, 2, DType>(bbox_weight_shape, s);

    // get input tensors
    // rois [n, 5] (batch_index, x1, y1, x2, y2)
    Tensor<xpu, 2, DType> all_rois = inputs[0].get<xpu, 2, DType>(s);
    // gt_boxes [n, 5] (x1, y1, x2, y2, cls)
    Tensor<xpu, 2, DType> all_gt_boxes = inputs[1].get<xpu, 2, DType>(s);

    // get destination tensors
    Tensor<xpu, 2, DType> out0 =
        outputs[0].get_with_shape<xpu, 2, DType>(rois_shape, s);
    Tensor<xpu, 1, DType> out1 =
        outputs[1].get_with_shape<xpu, 1, DType>(label_shape, s);
    Tensor<xpu, 2, DType> out2 =
        outputs[2].get_with_shape<xpu, 2, DType>(bbox_target_shape, s);
    Tensor<xpu, 2, DType> out3 =
        outputs[3].get_with_shape<xpu, 2, DType>(bbox_weight_shape, s);

    // Main logic
    CHECK_EQ(all_rois.CheckContiguous(), true);
    CHECK_EQ(all_gt_boxes.CheckContiguous(), true);
    CHECK_EQ(out0.CheckContiguous(), true);
    CHECK_EQ(out1.CheckContiguous(), true);
    CHECK_EQ(out2.CheckContiguous(), true);
    CHECK_EQ(out3.CheckContiguous(), true);

    ProposalTargetOpForwardImpl(params, all_rois, all_gt_boxes, out0, out1,
                                out2, out3);

    // copy results
    ASSIGN_DISPATCH(out0, req[0], rois);
    ASSIGN_DISPATCH(out1, req[1], labels);
    ASSIGN_DISPATCH(out2, req[2], bbox_targets);
    ASSIGN_DISPATCH(out3, req[3], bbox_weights);
  });
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
