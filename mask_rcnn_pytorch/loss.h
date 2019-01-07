#ifndef LOSS_H
#define LOSS_H

#include <torch/torch.h>

/* RPN anchor classifier loss.
 * rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
 *           -1=negative, 0=neutral anchor.
 * rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
 */
torch::Tensor ComputeRpnClassLoss(torch::Tensor rpn_match,
                                  torch::Tensor rpn_class_logits);

/* Return the RPN bounding box loss graph.
 * target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
 *   Uses 0 padding to fill in unsed bbox deltas.
 * rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
 *           -1=negative, 0=neutral anchor.
 * rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
 */
torch::Tensor ComputeRpnBBoxLoss(torch::Tensor target_bbox,
                                 torch::Tensor rpn_match,
                                 torch::Tensor rpn_bbox);

/* Loss for the classifier head of Mask RCNN.
 * target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
 *   padding to fill in the array.
 * pred_class_logits: [batch, num_rois, num_classes]
 */
torch::Tensor ComputeMrcnnClassLoss(torch::Tensor target_class_ids,
                                    torch::Tensor pred_class_logits);

/* Loss for Mask R-CNN bounding box refinement.
 * target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
 * target_class_ids: [batch, num_rois]. Integer class IDs.
 * pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
 */
torch::Tensor ComputeMrcnnBBoxLoss(torch::Tensor target_bbox,
                                   torch::Tensor target_class_ids,
                                   torch::Tensor pred_bbox);

/* Mask binary cross-entropy loss for the masks head.
 * target_masks: [batch, num_rois, height, width].
 *   A float32 tensor of values 0 or 1. Uses zero padding to fill array.
 * target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
 * pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
 *             with values from 0 to 1.
 */
torch::Tensor ComputeMrcnnMaskLoss(torch::Tensor target_masks,
                                   torch::Tensor target_class_ids,
                                   torch::Tensor pred_masks);

#endif  // LOSS_H
