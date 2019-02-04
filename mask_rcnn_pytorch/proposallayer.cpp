#include "proposallayer.h"
#include "boxutils.h"
#include "debug.h"
#include "imageutils.h"
#include "nms.h"
#include "nnutils.h"

at::Tensor ProposalLayer(std::vector<at::Tensor> inputs,
                         int64_t proposal_count,
                         float nms_threshold,
                         at::Tensor anchors,
                         const Config& config) {
  // Currently only supports batchsize 1
  inputs[0] = inputs[0].squeeze(0);
  inputs[1] = inputs[1].squeeze(0);

  // Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
  auto scores = inputs[0].narrow(1, 1, 1);

  // Box deltas [batch, num_rois, 4]
  auto deltas = inputs[1];

  auto std_dev =
      torch::tensor(config.rpn_bbox_std_dev,
                    at::TensorOptions().requires_grad(false).dtype(at::kFloat));
  if (config.gpu_count > 0)
    std_dev = std_dev.cuda();
  deltas = deltas * std_dev;

  // Improve performance by trimming to top anchors by score
  // and doing the rest on the smaller subset.
  auto pre_nms_limit = std::min(int64_t{6000}, anchors.size(0));
  at::Tensor order;
  std::tie(scores, order) = scores.sort(/*dim*/ 0, /*descending*/ true);

  order = order.narrow(0, 0, std::min(order.numel(), pre_nms_limit)).flatten();
  scores =
      scores.narrow(0, 0, std::min(scores.numel(), pre_nms_limit)).flatten();

  // TODO : (Legacy)Support batch size > 1 ff.
  deltas = deltas.index_select(0, order);
  anchors = anchors.index_select(0, order);

  // Apply deltas to anchors to get refined anchors.
  // [batch, N, (y1, x1, y2, x2)]
  auto boxes = ApplyBoxDeltas(anchors, deltas);

  // Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
  auto height = config.image_shape[0];
  auto width = config.image_shape[1];
  Window window{0, 0, height, width};
  boxes = ClipBoxes(boxes, window);

  // Filter out small boxes
  // According to Xinlei Chen's paper, this reduces detection accuracy
  // for small objects, so we're skipping it.

  // Non-max suppression
  auto keep = Nms(torch::cat({boxes, scores.unsqueeze(1)}, 1), nms_threshold);
  keep = keep.narrow(0, 0, std::min(keep.size(0), proposal_count));
  boxes = boxes.index_select(0, keep);

  // Normalize dimensions to range of 0 to 1.
  auto norm =
      torch::tensor({static_cast<float>(height), static_cast<float>(width),
                     static_cast<float>(height), static_cast<float>(width)},
                    at::requires_grad(false).dtype(at::kFloat));

  if (config.gpu_count > 0)
    norm = norm.cuda();
  auto normalized_boxes = boxes / norm;

  // Add back batch dimension
  normalized_boxes = normalized_boxes.unsqueeze(0);

  return normalized_boxes;
}
