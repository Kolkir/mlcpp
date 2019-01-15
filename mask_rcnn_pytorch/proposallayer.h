#ifndef PROPOSALLAYER_H
#define PROPOSALLAYER_H

#include "config.h"

#include <torch/torch.h>

/*
 *  Receives anchor scores and selects a subset to pass as proposals
 *   to the second stage. Filtering is done based on anchor scores and
 *  non-max suppression to remove overlaps. It also applies bounding
 *  box refinment details to anchors.
 *  Inputs:
 *      rpn_probs: [batch, anchors, (bg prob, fg prob)]
 *      rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
 *  Returns:
 *      Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
 */
at::Tensor ProposalLayer(std::vector<at::Tensor> inputs,
                         int64_t proposal_count,
                         float nms_threshold,
                         at::Tensor anchors,
                         const Config& config);

#endif  // PROPOSALLAYER_H
