#include "proposaltarget_op.h"
#include "proposaltarget_op.hpp"

namespace mxnet {
namespace op {

template <>
Operator* CreateOp<gpu>(ProposalTargetParam param) {
  return new ProposalTargetOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
