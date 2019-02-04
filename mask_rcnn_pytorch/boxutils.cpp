#include "boxutils.h"

/* Calculates IoU of the given box with the array of the given boxes.
 * box: 1D vector [y1, x1, y2, x2]
 * boxes: [boxes_count, (y1, x1, y2, x2)]
 * box_area: float. the area of 'box'
 * boxes_area: array of length boxes_count.
 * Note: the areas are passed in rather than calculated here for
 *       efficency. Calculate once in the caller to avoid duplicate work.
 */
static at::Tensor ComputeIou(at::Tensor box,
                             at::Tensor boxes,
                             at::Tensor box_area,
                             at::Tensor boxes_area) {
  auto y1 = torch::max(box[0], boxes.narrow(1, 0, 1));
  auto y2 = torch::min(box[2], boxes.narrow(1, 2, 1));
  auto x1 = torch::max(box[1], boxes.narrow(1, 1, 1));
  auto x2 = torch::min(box[3], boxes.narrow(1, 3, 1));
  auto zero_val = torch::tensor(0.f);
  if (box.is_cuda())
    zero_val = zero_val.cuda();
  auto intersection =
      torch::max(x2 - x1, zero_val) * torch::max(y2 - y1, zero_val);
  auto union_ = box_area + boxes_area - intersection;
  auto iou = intersection / union_;
  return iou;
}

torch::Tensor BBoxOverlapsLoops(torch::Tensor boxes1, torch::Tensor boxes2) {
  // Areas of anchors and GT boxes
  auto area1 = (boxes1.narrow(1, 2, 1) - boxes1.narrow(1, 0, 1)) *
               (boxes1.narrow(1, 3, 1) - boxes1.narrow(1, 1, 1));
  auto area2 = (boxes2.narrow(1, 2, 1) - boxes2.narrow(1, 0, 1)) *
               (boxes2.narrow(1, 3, 1) - boxes2.narrow(1, 1, 1));

  // Compute overlaps to generate matrix [boxes1 count, boxes2 count]
  // Each cell contains the IoU value.
  auto overlaps = torch::zeros({boxes1.size(0), boxes2.size(0)});
  if (boxes2.is_cuda())
    overlaps = overlaps.cuda();
  for (int64_t i = 0; i < overlaps.size(1); ++i) {
    auto box2 = boxes2[i];
    auto iou = ComputeIou(box2, boxes1, area2[i], area1);
    overlaps.narrow(1, i, 1) = iou;
  }
  return overlaps;
}

torch::Tensor BoxRefinement(torch::Tensor box, torch::Tensor gt_box) {
  auto height = box.narrow(1, 2, 1) - box.narrow(1, 0, 1);
  auto width = box.narrow(1, 3, 1) - box.narrow(1, 1, 1);
  auto center_y = box.narrow(1, 0, 1) + 0.5f * height;
  auto center_x = box.narrow(1, 1, 1) + 0.5f * width;

  auto gt_height = gt_box.narrow(1, 2, 1) - gt_box.narrow(1, 0, 1);
  auto gt_width = gt_box.narrow(1, 3, 1) - gt_box.narrow(1, 1, 1);
  auto gt_center_y = gt_box.narrow(1, 0, 1) + 0.5f * gt_height;
  auto gt_center_x = gt_box.narrow(1, 1, 1) + 0.5f * gt_width;

  auto dy = (gt_center_y - center_y) / height;
  auto dx = (gt_center_x - center_x) / width;
  auto dh = torch::log(gt_height / height);
  auto dw = torch::log(gt_width / width);

  auto result = torch::stack(
      {dy.squeeze(1), dx.squeeze(1), dh.squeeze(1), dw.squeeze(1)}, /*dim*/ 1);
  return result;
}

at::Tensor ApplyBoxDeltas(at::Tensor boxes, at::Tensor deltas) {
  // Convert to y, x, h, w
  auto height = boxes.narrow(1, 2, 1) - boxes.narrow(1, 0, 1);
  auto width = boxes.narrow(1, 3, 1) - boxes.narrow(1, 1, 1);
  auto center_y = boxes.narrow(1, 0, 1) + 0.5f * height;
  auto center_x = boxes.narrow(1, 1, 1) + 0.5f * width;
  // Apply deltas
  center_y += deltas.narrow(1, 0, 1) * height;
  center_x += deltas.narrow(1, 1, 1) * width;

  height *= torch::exp(deltas.narrow(1, 2, 1));
  width *= torch::exp(deltas.narrow(1, 3, 1));
  // Convert back to y1, x1, y2, x2
  auto y1 = center_y - 0.5f * height;
  auto x1 = center_x - 0.5f * width;
  auto y2 = y1 + height;
  auto x2 = x1 + width;

  auto result = torch::stack(
      {y1.squeeze(), x1.squeeze(), y2.squeeze(), x2.squeeze()}, /*dim*/ 1);
  return result;
}

/*
 * boxes: [N, 4] each col is y1, x1, y2, x2
 * window: [4] in the form y1, x1, y2, x2
 */
at::Tensor ClipBoxes(at::Tensor boxes, Window window) {
  boxes = torch::stack(
      {boxes.narrow(1, 0, 1)
           .clamp(static_cast<float>(window.y1), static_cast<float>(window.y2))
           .squeeze(),
       boxes.narrow(1, 1, 1)
           .clamp(static_cast<float>(window.x1), static_cast<float>(window.x2))
           .squeeze(),
       boxes.narrow(1, 2, 1)
           .clamp(static_cast<float>(window.y1), static_cast<float>(window.y2))
           .squeeze(),
       boxes.narrow(1, 3, 1)
           .clamp(static_cast<float>(window.x1), static_cast<float>(window.x2))
           .squeeze()},
      1);
  return boxes;
}

at::Tensor ClipToWindow(const Window& window, at::Tensor boxes) {
  boxes.narrow(1, 0, 1) = boxes.narrow(1, 0, 1).clamp(
      static_cast<float>(window.y1), static_cast<float>(window.y2));
  boxes.narrow(1, 1, 1) = boxes.narrow(1, 1, 1).clamp(
      static_cast<float>(window.x1), static_cast<float>(window.x2));
  boxes.narrow(1, 2, 1) = boxes.narrow(1, 2, 1).clamp(
      static_cast<float>(window.y1), static_cast<float>(window.y2));
  boxes.narrow(1, 3, 1) = boxes.narrow(1, 3, 1).clamp(
      static_cast<float>(window.x1), static_cast<float>(window.x2));
  return boxes;
}
