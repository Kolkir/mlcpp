#include "nms.h"
#include "debug.h"

#include "nms/nms.h"
#include "nms/nms_cuda.h"

at::Tensor Nms(at::Tensor dets, float thresh) {
  auto scores = dets.narrow(1, 4, 1);
  at::Tensor order;
  std::tie(std::ignore, order) = scores.sort(0, /*descending*/ true);
  auto keep = torch::full({dets.size(0)}, 0, at::dtype(at::kLong));
  auto num_out = torch::full({1}, 0, at::dtype(at::kLong));

  if (!dets.is_cuda()) {
    auto x1 = dets.narrow(1, 1, 1);
    auto y1 = dets.narrow(1, 0, 1);
    auto x2 = dets.narrow(1, 3, 1);
    auto y2 = dets.narrow(1, 2, 1);
    auto areas = (x2 - x1 + 1) * (y2 - y1 + 1);
    cpu_nms(keep, num_out, dets, order, areas, thresh);
    return keep.narrow(0, 0, num_out.item<int64_t>());
  } else {
    auto dets_temp = torch::full(dets.sizes(), 0, at::dtype(at::kFloat)).cuda();
    dets_temp.narrow(1, 0, 1) = dets.narrow(1, 1, 1);
    dets_temp.narrow(1, 1, 1) = dets.narrow(1, 0, 1);
    dets_temp.narrow(1, 2, 1) = dets.narrow(1, 3, 1);
    dets_temp.narrow(1, 3, 1) = dets.narrow(1, 2, 1);
    dets_temp.narrow(1, 4, 1) = dets.narrow(1, 4, 1);
    dets = dets.index(order).contiguous();
    gpu_nms(keep, num_out, dets_temp, thresh);
    auto ind = keep.narrow(0, 0, num_out.item<int64_t>()).cuda();
    return order.take(ind).contiguous();
  }
}
