#ifndef DETECTIONTARGETLAYER_H
#define DETECTIONTARGETLAYER_H

#include "config.h"
#include "imageutils.h"

#include <torch/torch.h>

/* Subsamples proposals and generates target box refinment, class_ids,
 * and masks for each.
 * Inputs:
 * proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
 *             be zero padded if there are not enough proposals.
 *  gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
 *  gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
 *            coordinates.
 *  gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type
 *
 *  Returns: Target ROIs and corresponding class IDs, bounding box shifts,
 *  and masks.
 *  rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
 *        coordinates
 *  target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
 *  target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
 *                  (dy, dx, log(dh), log(dw), class_id)]
 *                 Class-specific bbox refinments.
 *  target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
 *               Masks cropped to bbox boundaries and resized to neural
 *               network output size.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> DetectionTargetLayer(
    const Config& config,
    at::Tensor proposals,
    at::Tensor gt_class_ids,
    at::Tensor gt_boxes,
    at::Tensor gt_masks);

#endif  // DETECTIONTARGETLAYER_H
