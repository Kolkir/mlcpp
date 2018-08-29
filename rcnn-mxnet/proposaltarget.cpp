#include "proposaltarget.h"

#include <elemwise_op_common.h>

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
