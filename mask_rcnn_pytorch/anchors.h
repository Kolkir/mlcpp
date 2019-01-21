#ifndef ANCHORS_H
#define ANCHORS_H

#include <torch/torch.h>

#include <stdint.h>
#include <vector>

/*
 * Generate anchors at different levels of a feature pyramid. Each scale
 * is associated with a level of the pyramid, but each ratio is used in
 * all levels of the pyramid.
 * Returns:
 * anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
 *     with the same order of the given scales. So, anchors of scale[0] come
 *     first, then anchors of scale[1], and so on.
 */
torch::Tensor GeneratePyramidAnchors(
    std::vector<float> scales_vec,
    std::vector<float> ratios_vec,
    const std::vector<std::pair<float, float>>& feature_shapes,
    const std::vector<float>& feature_strides,
    float anchor_stride);

#endif  // ANCHORS_H
