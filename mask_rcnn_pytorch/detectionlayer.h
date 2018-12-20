#ifndef DETECTIONLAYER_H
#define DETECTIONLAYER_H

#include "config.h"
#include "imageutils.h"

#include <torch/torch.h>
/*
 * Takes classified proposal boxes and their bounding box deltas and
 * returns the final detection boxes.
 * Returns:
 * [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
 */

at::Tensor DetectionLayer(const Config& config,
                          at::Tensor rois,
                          at::Tensor probs,
                          at::Tensor deltas,
                          const std::vector<ImageMeta>& image_meta);

#endif  // DETECTIONLAYER_H
