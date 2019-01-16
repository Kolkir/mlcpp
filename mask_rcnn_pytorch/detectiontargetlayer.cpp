#include "detectiontargetlayer.h"
#include "boxutils.h"
#include "nnutils.h"
#include "roialign/crop_and_resize.h"
#include "roialign/crop_and_resize_gpu.h"

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> DetectionTargetLayer(
    const Config& config,
    at::Tensor proposals,
    at::Tensor gt_class_ids,
    at::Tensor gt_boxes,
    at::Tensor gt_masks) {
  // Currently only supports batchsize 1
  proposals = proposals.squeeze(0);
  gt_class_ids = gt_class_ids.squeeze(0);
  gt_boxes = gt_boxes.squeeze(0);
  gt_masks = gt_masks.squeeze(0);

  //  Handle COCO crowds
  //  A crowd box in COCO is a bounding box around several instances. Exclude
  //  them from training. A crowd box is given a negative class ID.
  //  Now they are excluded in coco loader

  // Compute overlaps matrix [proposals, gt_boxes]
  auto overlaps = BBoxOverlaps(proposals, gt_boxes);

  // Determine postive and negative ROIs
  auto roi_iou_max = std::get<0>(torch::max(overlaps, /*dim*/ 1));

  // 1. Positive ROIs are those with >= 0.5 IoU with a GT box
  auto positive_roi_bool = roi_iou_max >= 0.5f;

  at::Tensor roi_gt_class_ids;
  at::Tensor deltas;
  at::Tensor masks;

  // Subsample ROIs. Aim for 33% positive
  // Positive ROIs
  int64_t positive_count = 0;
  torch::Tensor positive_rois;
  auto positive_indices = torch::nonzero(positive_roi_bool);
  if (!is_empty(positive_indices)) {
    positive_indices = positive_indices.narrow(1, 0, 1);

    positive_count =
        int(config.train_rois_per_image * config.roi_positive_ratio);
    auto rand_idx =
        torch::randperm(positive_indices.size(0), at::dtype(at::kLong));
    rand_idx = rand_idx.narrow(
        0, 0, std::min(positive_indices.size(0), positive_count));
    if (config.gpu_count > 0) {
      rand_idx = rand_idx.cuda();
    }
    positive_indices = positive_indices.take(rand_idx);
    positive_count = positive_indices.size(0);
    positive_rois = proposals.index_select(0, positive_indices);

    //   Assign positive ROIs to GT boxes.
    auto positive_overlaps = overlaps.index_select(0, positive_indices);
    auto roi_gt_box_assignment =
        std::get<1>(torch::max(positive_overlaps, /*dim*/ 1));
    auto roi_gt_boxes = gt_boxes.index_select(0, roi_gt_box_assignment);
    roi_gt_class_ids = gt_class_ids.take(roi_gt_box_assignment);

    //   Compute bbox refinement for positive ROIs
    {
      auto deltas_data = BoxRefinement(positive_rois, roi_gt_boxes);
      deltas =
          torch::from_blob(deltas_data.cpu().data<float>(), deltas_data.sizes(),
                           at::dtype(at::kFloat).requires_grad(false))
              .clone();
    }
    if (config.gpu_count > 0)
      deltas = deltas.cuda();

    auto std_dev = torch::tensor(config.rpn_bbox_std_dev,
                                 at::dtype(at::kFloat).requires_grad(false));
    if (config.gpu_count > 0)
      std_dev = std_dev.cuda();
    deltas /= std_dev;

    //   Assign positive ROIs to GT masks
    auto roi_masks = gt_masks.index_select(
        0, roi_gt_box_assignment);  //[roi_gt_box_assignment,:,:]

    //   Compute mask targets
    auto boxes = positive_rois;
    if (config.use_mini_mask) {
      // Transform ROI corrdinates from normalized image space
      // to normalized mini-mask space.
      auto yxyx = positive_rois.chunk(4, /*dim*/ 1);
      auto y1 = yxyx[0];
      auto x1 = yxyx[1];
      auto y2 = yxyx[2];
      auto x2 = yxyx[3];
      auto gyxyx = roi_gt_boxes.chunk(4, /*dim*/ 1);
      auto gt_y1 = gyxyx[0];
      auto gt_x1 = gyxyx[1];
      auto gt_y2 = gyxyx[2];
      auto gt_x2 = gyxyx[3];
      auto gt_h = gt_y2 - gt_y1;
      auto gt_w = gt_x2 - gt_x1;
      y1 = (y1 - gt_y1) / gt_h;
      x1 = (x1 - gt_x1) / gt_w;
      y2 = (y2 - gt_y1) / gt_h;
      x2 = (x2 - gt_x1) / gt_w;
      boxes = torch::cat({y1, x1, y2, x2}, /*dim*/ 1);
    }
    auto box_ids = torch::arange(roi_masks.size(0),
                                 at::requires_grad(false).dtype(at::kInt));
    if (config.gpu_count > 0)
      box_ids = box_ids.cuda();
    masks = torch::zeros({}, at::dtype(at::kFloat).requires_grad(false));
    if (config.gpu_count > 0)
      masks = masks.cuda();

    if (config.gpu_count > 0) {
      crop_and_resize_gpu_forward(roi_masks.unsqueeze(1), boxes, box_ids, 0,
                                  config.mask_shape[0], config.mask_shape[1],
                                  masks);
    } else {
      crop_and_resize_forward(roi_masks.unsqueeze(1), boxes, box_ids, 0,
                              config.mask_shape[0], config.mask_shape[1],
                              masks);
    }
    masks = masks.squeeze(1);

    //  Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    //   binary cross entropy loss.
    masks = torch::round(masks);
  } else {
    positive_count = 0;
  }

  //  2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
  auto negative_roi_bool = roi_iou_max < 0.5f;
  // negative_roi_bool = negative_roi_bool & no_crowd_bool
  // Negative ROIs. Add enough to maintain positive:negative ratio.
  int64_t negative_count = 0;
  torch::Tensor negative_rois;
  if (!is_empty(torch::nonzero(negative_roi_bool)) && positive_count > 0) {
    auto negative_indices = torch::nonzero(negative_roi_bool).narrow(1, 0, 1);
    auto r = 1.0f / config.roi_positive_ratio;
    negative_count = static_cast<int64_t>(r * positive_count - positive_count);
    auto rand_idx =
        torch::randperm(negative_indices.size(0), at::dtype(at::kLong));
    rand_idx = rand_idx.narrow(
        0, 0, std::min(negative_indices.size(0), negative_count));
    if (config.gpu_count > 0)
      rand_idx = rand_idx.cuda();
    negative_indices = negative_indices.take(rand_idx);
    negative_count = negative_indices.size(0);
    negative_rois = proposals.index_select(0, negative_indices);
  } else {
    negative_count = 0;
  }

  // Append negative ROIs and pad bbox deltas and masks that
  // are not used for negative ROIs with zeros.
  at::Tensor rois;
  if (positive_count > 0 && negative_count > 0) {
    rois = torch::cat({positive_rois, negative_rois}, /*dim*/ 0);
    auto zeros = torch::zeros({negative_count},
                              at::requires_grad(false).dtype(at::kInt));
    if (config.gpu_count > 0)
      zeros = zeros.cuda();
    roi_gt_class_ids = torch::cat({roi_gt_class_ids, zeros}, /*dim*/ 0);
    zeros = torch::zeros({negative_count, 4}, at::requires_grad(false));
    if (config.gpu_count > 0)
      zeros = zeros.cuda();
    deltas = torch::cat({deltas, zeros}, /*dim*/ 0);
    zeros = torch::zeros(
        {negative_count, config.mask_shape[0], config.mask_shape[1]},
        at::requires_grad(false));
    if (config.gpu_count > 0)
      zeros = zeros.cuda();
    masks = torch::cat({masks, zeros}, /*dim*/ 0);
  } else if (positive_count > 0) {
    rois = positive_rois;
  } else if (negative_count > 0) {
    rois = negative_rois;
    auto zeros = torch::zeros({negative_count}, at::requires_grad(false));
    if (config.gpu_count > 0)
      zeros = zeros.cuda();
    roi_gt_class_ids = zeros;
    zeros = torch::zeros({negative_count, 4},
                         at::requires_grad(false).dtype(at::kInt));
    if (config.gpu_count > 0)
      zeros = zeros.cuda();
    deltas = zeros;
    zeros = torch::zeros(
        {negative_count, config.mask_shape[0], config.mask_shape[1]},
        at::requires_grad(false));
    if (config.gpu_count > 0)
      zeros = zeros.cuda();
    masks = zeros;
  } else {
    rois = torch::empty({}, at::dtype(at::kFloat).requires_grad(false));
    roi_gt_class_ids =
        torch::empty({}, at::dtype(at::kFloat).requires_grad(false));
    deltas = torch::empty({}, at::dtype(at::kFloat).requires_grad(false));
    masks = torch::empty({}, at::dtype(at::kFloat).requires_grad(false));
    if (config.gpu_count) {
      rois = rois.cuda();
      roi_gt_class_ids = roi_gt_class_ids.cuda();
      deltas = deltas.cuda();
      masks = masks.cuda();
    }
  }
  return {rois, roi_gt_class_ids, deltas, masks};
}
