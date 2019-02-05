#include "loss.h"
#include "imageutils.h"
#include "nnutils.h"

at::Tensor ComputeRpnClassLoss(at::Tensor rpn_match,
                               at::Tensor rpn_class_logits) {
  // Squeeze last dim to simplify
  rpn_match = rpn_match.squeeze(2);

  // Get anchor classes. Convert the -1/+1 match to 0/1 values.
  auto anchor_class = (rpn_match == 1).to(at::dtype(at::kLong));

  // Positive and Negative anchors contribute to the loss,
  // but neutral anchors (match value = 0) don't.
  auto indices = torch::nonzero(rpn_match != 0);

  // Pick rows that contribute to the loss and filter out the rest.
  auto y_ind = indices.narrow(1, 0, 1).squeeze();
  auto x_ind = indices.narrow(1, 1, 1).squeeze();

  rpn_class_logits = rpn_class_logits.index({y_ind, x_ind});

  anchor_class = anchor_class.index({y_ind, x_ind});

  // Crossentropy loss
  auto loss = torch::nll_loss(rpn_class_logits.log_softmax(1), anchor_class);
  return loss;
}

at::Tensor ComputeRpnBBoxLoss(at::Tensor target_bbox,
                              at::Tensor rpn_match,
                              at::Tensor rpn_bbox) {
  // Squeeze last dim to simplify
  rpn_match = rpn_match.squeeze(2);

  // Positive anchors contribute to the loss, but negative and
  // neutral anchors (match value of 0 or -1) don't.
  auto indices = torch::nonzero(rpn_match == 1);

  // Pick bbox deltas that contribute to the loss
  auto y_ind = indices.narrow(1, 0, 1).squeeze();
  auto x_ind = indices.narrow(1, 1, 1).squeeze();

  rpn_bbox = rpn_bbox.index({y_ind, x_ind});

  // Trim target bounding box deltas to the same length as rpn_bbox.
  target_bbox = target_bbox[0].narrow(0, 0, rpn_bbox.size(0));

  // Smooth L1 loss
  auto box_target = torch::broadcast_tensors({rpn_bbox, target_bbox});
  auto loss = torch::smooth_l1_loss(box_target[0], box_target[1]);

  return loss;
}

at::Tensor ComputeMrcnnClassLoss(at::Tensor target_class_ids,
                                 at::Tensor pred_class_logits) {
  at::Tensor loss;
  if (!is_empty(target_class_ids)) {
    loss = torch::nll_loss(pred_class_logits.log_softmax(1),
                           target_class_ids.to(at::dtype(at::kLong)));
  } else {
    loss = torch::tensor(0.f, at::dtype(at::kFloat).requires_grad(false));
    if (target_class_ids.is_cuda())
      loss = loss.cuda();
  }

  return loss;
}

at::Tensor ComputeMrcnnBBoxLoss(at::Tensor target_bbox,
                                at::Tensor target_class_ids,
                                at::Tensor pred_bbox) {
  at::Tensor loss;
  if (!is_empty(target_class_ids)) {
    // Only positive ROIs contribute to the loss. And only
    // the right class_id of each ROI. Get their indicies.
    auto positive_ix = torch::nonzero(target_class_ids > 0).narrow(1, 0, 1);
    auto positive_class_ids =
        target_class_ids.take(positive_ix).to(at::dtype(at::kLong));
    auto indices = torch::stack({positive_ix, positive_class_ids}, /*dim*/ 1);

    // Gather the masks (predicted and true) that contribute to loss
    auto y_ind = indices.narrow(1, 0, 1).squeeze();
    auto x_ind = indices.narrow(1, 1, 1).squeeze();

    // Gather the deltas (predicted and true) that contribute to loss
    target_bbox = target_bbox.index(y_ind);
    pred_bbox = pred_bbox.index({y_ind, x_ind});

    // Smooth L1 loss
    auto box_target = torch::broadcast_tensors({pred_bbox, target_bbox});
    loss = torch::smooth_l1_loss(box_target[0], box_target[1]);
  } else {
    loss = torch::tensor(0.f, at::dtype(at::kFloat).requires_grad(false));
    if (target_class_ids.is_cuda())
      loss = loss.cuda();
  }
  return loss;
}

at::Tensor ComputeMrcnnMaskLoss(at::Tensor target_masks,
                                at::Tensor target_class_ids,
                                at::Tensor pred_masks) {
  at::Tensor loss;
  if (!is_empty(target_class_ids)) {
    // Only positive ROIs contribute to the loss. And only
    // the class specific mask of each ROI.
    auto positive_ix = torch::nonzero(target_class_ids > 0).narrow(1, 0, 1);
    auto positive_class_ids =
        target_class_ids.take(positive_ix).to(at::dtype(at::kLong));
    auto indices = torch::stack({positive_ix, positive_class_ids}, /*dim*/ 1);

    // Gather the masks (predicted and true) that contribute to loss
    auto y_ind = indices.narrow(1, 0, 1).squeeze();
    auto x_ind = indices.narrow(1, 1, 1).squeeze();

    auto y_true = target_masks.index(y_ind);
    auto y_pred = pred_masks.index({y_ind, x_ind});

    // Binary cross entropy
    loss = torch::binary_cross_entropy(y_pred, y_true);
  } else {
    loss = torch::tensor(0.f, at::dtype(at::kFloat).requires_grad(false));
    if (target_class_ids.is_cuda())
      loss = loss.cuda();
  }
  return loss;
}
