#include "proposaltarget.h"

#include <elemwise_op_common.h>
#include <Eigen/Dense>

namespace mshadow {
template <typename Dtype>
inline void ProposalTargetOpForwardImpl(
    const mxnet::op::ProposalTargetParam& params,
    const Tensor<cpu, 2, Dtype>& all_rois_t,
    const Tensor<cpu, 2, Dtype>& all_gt_boxes_t,
    const Tensor<cpu, 2, Dtype>& b_rois_t,
    const Tensor<cpu, 1, Dtype>& b_labels_t,
    const Tensor<cpu, 2, Dtype>& b_bbox_targets_t,
    const Tensor<cpu, 2, Dtype>& b_bbox_weights_t) {
  using Matrix = Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic>;

  Eigen::Map<Matrix> all_rois(all_rois_t.dptr_, all_rois_t.size(0),
                              all_rois_t.size(1));

  Eigen::ArrayXf all_rois_idns =
      Eigen::ArrayXf::Constant(all_rois_t.size(0), 1.f);
  for (int batch_idx = 0; batch_idx < params.batch_images; ++batch_idx) {
    Eigen::ArrayXf rois_b_inds = (all_rois.col(0).array() == batch_idx)
                                     .select(all_rois_idns, -all_rois_idns);
    auto num_b_rois = (rois_b_inds > 0).count();
  }
}
}  // namespace mshadow

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(ProposalTargetParam);

NNVM_REGISTER_OP(proposal_target)
    .describe(R"code(ProposalTarget)code" ADD_FILELINE)
    .set_attr_parser(ParamParser<ProposalTargetParam>)
    .set_num_inputs(2)
    .set_num_outputs(4)
    .set_attr<nnvm::FListInputNames>(
        "FListInputNames",
        [](const NodeAttrs& /*attrs*/) {
          return std::vector<std::string>{"rois", "gt_boxes"};
        })
    .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                      [](const NodeAttrs& /*attrs*/) {
                                        return std::vector<std::string>{
                                            "rois_output", "label",
                                            "bbox_target", "bbox_weight"};
                                      })
    .set_attr<nnvm::FInferShape>("FInferShape", ProposalTargetOpShape)
    .set_attr<nnvm::FInferType>("FInferType", ProposalTargetOpType)
    .set_attr<FCompute>("FCompute<cpu>", ProposalTargetOpForward<cpu>)
    .set_attr<nnvm::FGradient>("FGradient",
                               ElemwiseGradUseIn{"_backward_proposal_target"})
    .add_argument("rois", "NDArray-or-Symbol", "Input ndarray")
    .add_argument("gt_boxes", "NDArray-or-Symbol", "Input ndarray")
    .add_arguments(ProposalTargetParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_proposal_target)
    .set_attr_parser(ParamParser<ProposalTargetParam>)
    .set_num_inputs(2)
    .set_num_outputs(2)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FCompute>("FCompute<cpu>", ProposalTargetOpBackward<cpu>);
}  // namespace op
}  // namespace mxnet
