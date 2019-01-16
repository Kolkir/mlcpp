#include "anchors.h"
#include "debug.h"

#include <iostream>

namespace {

/*
 *   scale: anchor size in pixels. Example: [32, 64, 128]
 *   ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
 *   shape: [height, width] spatial shape of the feature map over which
 *           to generate anchors.
 *   feature_stride: Stride of the feature map relative to the image in pixels.
 *   anchor_stride: Stride of anchors on the feature map. For example, if the
 *       value is 2 then generate anchors for every other feature map pixel.
 */
torch::Tensor GenerateAnchors(torch::Tensor scales,
                              torch::Tensor ratios,
                              const std::pair<float, float>& shape,
                              float feature_stride,
                              float anchor_stride) {
  // Get all combinations of scales and ratios
  auto mesh = torch::meshgrid({scales, ratios});
  scales = mesh[0].flatten();
  ratios = mesh[1].flatten();

  // Enumerate heights and widths from scales and ratios
  auto heights = scales / torch::sqrt(ratios);
  auto widths = scales * torch::sqrt(ratios);

  // Enumerate shifts in feature space
  auto shifts_y = torch::arange(0, shape.first, anchor_stride) * feature_stride;
  auto shifts_x =
      torch::arange(0, shape.second, anchor_stride) * feature_stride;

  mesh = torch::meshgrid({shifts_x, shifts_y});
  shifts_x = mesh[0].permute({1, 0}).flatten();
  shifts_y = mesh[1].permute({1, 0}).flatten();

  // Enumerate combinations of shifts, widths, and heights
  mesh = torch::meshgrid({widths, shifts_x});
  auto box_widths = mesh[0].permute({1, 0});
  auto box_centers_x = mesh[1].permute({1, 0});

  mesh = torch::meshgrid({heights, shifts_y});
  auto box_heights = mesh[0].permute({1, 0});
  auto box_centers_y = mesh[1].permute({1, 0});

  // Reshape to get a list of (y, x) and a list of (h, w)
  auto box_centers =
      torch::stack({box_centers_y, box_centers_x}, /*dim*/ 2).reshape({-1, 2});
  auto box_sizes =
      torch::stack({box_heights, box_widths}, /*dim*/ 2).reshape({-1, 2});

  // Convert to corner coordinates (y1, x1, y2, x2)
  auto boxes =
      torch::cat({box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes},
                 /*dim*/ 1);
  return boxes;
}
}  // namespace

torch::Tensor GeneratePyramidAnchors(
    std::vector<float> scales_vec,
    std::vector<float> ratios_vec,
    const std::vector<std::pair<float, float> >& feature_shapes,
    const std::vector<float>& feature_strides,
    float anchor_stride) {
  std::vector<at::Tensor> anchors;
  auto ratios = torch::tensor(ratios_vec);

  for (size_t i = 0; i < scales_vec.size(); ++i) {
    auto scale = torch::tensor(scales_vec[i]);
    anchors.push_back(GenerateAnchors(scale, ratios, feature_shapes[i],
                                      feature_strides[i], anchor_stride));
  }
  return at::cat(anchors, /*dim*/ 0);
}
