#include "proposaltarget_op.h"
#include "proposaltarget_op.hpp"

namespace mxnet {
namespace op {

template <>
Operator* CreateOp<cpu>(ProposalTargetParam param) {
  return new ProposalTargetOp<cpu>(param);
}

Operator* ProposalTargetProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ProposalTargetParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_proposal_target, ProposalTargetProp)
    .describe("Proposal Target Operator")
    .add_argument("rois", "NDArray-or-Symbol", "Input ndarray")
    .add_argument("gt_boxes", "NDArray-or-Symbol", "Input ndarray")
    .add_arguments(ProposalTargetParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
