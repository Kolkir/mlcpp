#ifndef ROIALIGN_H
#define ROIALIGN_H

#include <torch/torch.h>

/*
 *  Implements ROI Pooling on multiple levels of the feature pyramid.
 *   Params:
 *  - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
 *  - image_shape: [height, width, channels]. Shape of input image in pixels
 *  Inputs:
 *  - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
 *           coordinates.
 *  - Feature maps: List of feature maps from different levels of the pyramid.
 *                  Each is [batch, channels, height, width]
 *  Output:
 *  Pooled regions in the shape: [num_boxes, height, width, channels].
 *  The width and height are those specific in the pool_shape in the layer
 *  constructor.
 */
torch::Tensor PyramidRoiAlign(std::vector<torch::Tensor> input,
                              uint32_t pool_size,
                              const std::vector<int32_t>& image_shape);

#endif  // ROIALIGN_H
