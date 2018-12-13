#include "proposallayer.h"
#include "imageutils.h"
#include "nms.h"
#include "nnutils.h"

namespace {
/*
 * Applies the given deltas to the given boxes.
 * boxes: [N, 4] where each row is y1, x1, y2, x2
 * deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
 */
at::Tensor ApplyBoxDeltas(at::Tensor boxes, at::Tensor deltas) {
  // Convert to y, x, h, w
  auto height = boxes.narrow(1, 2, 1) - boxes.narrow(1, 0, 1);
  auto width = boxes.narrow(1, 3, 1) - boxes.narrow(1, 1, 1);
  auto center_y = boxes.narrow(1, 0, 1) + 0.5 * height;
  auto center_x = boxes.narrow(1, 1, 1) + 0.5 * width;
  // Apply deltas
  center_y += deltas.narrow(1, 0, 1) * height;
  center_x += deltas.narrow(1, 1, 1) * width;
  height *= torch::exp(deltas.narrow(1, 2, 1));
  width *= torch::exp(deltas.narrow(1, 3, 1));
  // Convert back to y1, x1, y2, x2
  auto y1 = center_y - 0.5 * height;
  auto x1 = center_x - 0.5 * width;
  auto y2 = y1 + height;
  auto x2 = x1 + width;
  auto result = torch::stack({y1, x1, y2, x2}, /*dim*/ 1);
  return result;
}

/*
 * boxes: [N, 4] each col is y1, x1, y2, x2
 * window: [4] in the form y1, x1, y2, x2
 */
at::Tensor ClipBoxes(at::Tensor boxes, Window window) {
  boxes =
      torch::stack({boxes.narrow(1, 0, 1).clamp(static_cast<float>(window.x1),
                                                static_cast<float>(window.x2)),
                    boxes.narrow(1, 1, 1).clamp(static_cast<float>(window.y1),
                                                static_cast<float>(window.y2)),
                    boxes.narrow(1, 2, 1).clamp(static_cast<float>(window.x1),
                                                static_cast<float>(window.x2)),
                    boxes.narrow(1, 3, 1).clamp(static_cast<float>(window.y1),
                                                static_cast<float>(window.y2))},
                   1);
  return boxes;
}

}  // namespace

at::Tensor ProposalLayer(std::vector<at::Tensor> inputs,
                         uint32_t proposal_count,
                         float nms_threshold,
                         at::Tensor anchors,
                         std::shared_ptr<const Config> config) {
  // Currently only supports batchsize 1
  inputs[0] = inputs[0].squeeze(0);
  inputs[1] = inputs[1].squeeze(0);

  // Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
  auto scores = inputs[0].narrow(1, 1, 1);  //[:, 1];

  // Box deltas [batch, num_rois, 4]
  auto deltas = inputs[1];
  auto std_dev =
      torch::tensor(config->rpn_bbox_std_dev,
                    at::TensorOptions().requires_grad(false).dtype(at::kFloat));
  if (config->gpu_count > 0)
    std_dev = std_dev.cuda();
  deltas = deltas * std_dev;

  // Improve performance by trimming to top anchors by score
  // and doing the rest on the smaller subset.
  auto pre_nms_limit = std::min(int64_t{6000}, anchors.size(0));
  at::Tensor order;
  std::tie(scores, order) = scores.sort(/*dim*/ -1, /*descending*/ true);
  order = order.narrow(1, pre_nms_limit, 1);    //[:pre_nms_limit];
  scores = scores.narrow(1, pre_nms_limit, 1);  //[:pre_nms_limit];
  //[order.data, :];  // TODO : Support batch size > 1 ff.
  deltas = deltas.index_select(0, order);
  anchors = anchors.index_select(0, order);  //[order.data, :];

  // Apply deltas to anchors to get refined anchors.
  // [batch, N, (y1, x1, y2, x2)]
  auto boxes = ApplyBoxDeltas(anchors, deltas);

  // Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
  auto height = config->image_shape[2];
  auto width = config->image_shape[3];
  Window window{0, 0, height, width};
  boxes = ClipBoxes(boxes, window);

  // Filter out small boxes
  // According to Xinlei Chen's paper, this reduces detection accuracy
  // for small objects, so we're skipping it.

  // Non-max suppression
  auto keep = Nms(torch::cat({boxes, scores.unsqueeze(1)}, 1), nms_threshold);
  keep = keep.narrow(1, proposal_count, 1);  //[:proposal_count];
  boxes = boxes.index_select(1, keep);       //[keep, :];

  // Normalize dimensions to range of 0 to 1.
  auto norm =
      torch::tensor({static_cast<float>(height), static_cast<float>(width),
                     static_cast<float>(height), static_cast<float>(width)},
                    at::requires_grad(false).dtype(at::kFloat));

  if (config->gpu_count > 0)
    norm = norm.cuda();
  auto normalized_boxes = boxes / norm;

  // Add back batch dimension
  normalized_boxes = normalized_boxes.unsqueeze(0);

  return normalized_boxes;
}
