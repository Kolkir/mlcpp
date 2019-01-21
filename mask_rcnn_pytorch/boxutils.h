#ifndef BOXUTILS_H
#define BOXUTILS_H

#include "imageutils.h"

#include <torch/torch.h>

/* Computes IoU overlaps between two sets of boxes.
 * boxes1, boxes2: [N, (y1, x1, y2, x2)].
 * !!! Requires a lot of memory - but without loops
 */
torch::Tensor BBoxOverlaps(torch::Tensor boxes1, torch::Tensor boxes2);

/* Computes IoU overlaps between two sets of boxes.
 * boxes1, boxes2: [N, (y1, x1, y2, x2)].
 * For better performance, pass the largest set first and the smaller second.
 */
torch::Tensor BBoxOverlapsLoops(torch::Tensor boxes1, torch::Tensor boxes2);

/* Compute refinement needed to transform box to gt_box.
 * box and gt_box are [N, (y1, x1, y2, x2)]
 */
torch::Tensor BoxRefinement(torch::Tensor box, torch::Tensor gt_box);

/*
 * Applies the given deltas to the given boxes.
 * boxes: [N, 4] where each row is y1, x1, y2, x2
 * deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
 */
at::Tensor ApplyBoxDeltas(at::Tensor boxes, at::Tensor deltas);

/*
 * boxes: [N, 4] each col is y1, x1, y2, x2
 * window: [4] in the form y1, x1, y2, x2
 */
at::Tensor ClipBoxes(at::Tensor boxes, Window window);

/*
 *    window: (y1, x1, y2, x2). The window in the image we want to clip to.
 *    boxes: [N, (y1, x1, y2, x2)]
 */
at::Tensor ClipToWindow(const Window& window, at::Tensor boxes);

#endif  // BOXUTILS_H
