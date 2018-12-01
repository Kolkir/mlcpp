#include "catch.hpp"

#include "../anchors.h"
#include "../config.h"

TEST_CASE("Generate anchors no throw", "[anchors]") {
  Config config;
  torch::Tensor boxes;
  CHECK_NOTHROW(boxes = GeneratePyramidAnchors(
                    config.rpn_anchor_scales, config.rpn_anchor_ratios,
                    config.backbone_shapes, config.backbone_strides,
                    config.rpn_anchor_stride));
}

TEST_CASE("Generate anchors", "[anchors]") {
  std::vector<uint32_t> image_shape = {800, 600};
  std::vector<float> scales = {32, 64};
  std::vector<float> ratios = {0.5f, 1.f, 2.f};
  std::vector<float> feature_strides = {4, 8};
  std::vector<std::pair<float, float>> feature_shapes;
  uint32_t anchor_stride = 10;

  // compute backbone size from input image size
  for (auto stride : feature_strides) {
    feature_shapes.push_back(
        {image_shape[0] / stride, image_shape[1] / stride});
  }

  uint64_t total_size = 0;
  for (size_t i = 0; i < scales.size(); ++i) {
    auto features_num = std::round(feature_shapes[i].first / anchor_stride) *
                        std::round(feature_shapes[i].second / anchor_stride);
    total_size += ratios.size() * features_num;
  }

  torch::Tensor boxes;
  CHECK_NOTHROW(boxes = GeneratePyramidAnchors(scales, ratios, feature_shapes,
                                               feature_strides, anchor_stride));
  REQUIRE(total_size == boxes.size(0));
  REQUIRE(4 == boxes.size(1));

  auto half_w = static_cast<float>(image_shape[0]) / 2.f;
  auto half_h = static_cast<float>(image_shape[1]) / 2.f;

  auto boxes_data = boxes.accessor<float, 2>();
  for (int64_t i = 0; i < boxes.size(0); ++i) {
    auto y1 = boxes_data[i][0];
    auto x1 = boxes_data[i][1];
    auto y2 = boxes_data[i][2];
    auto x2 = boxes_data[i][3];
    REQUIRE(x1 >= -half_h);
    REQUIRE(x2 >= -half_h);
    REQUIRE(y1 >= -half_w);
    REQUIRE(y2 >= -half_w);

    REQUIRE(x1 < static_cast<float>(image_shape[1]) + half_h);
    REQUIRE(x2 < static_cast<float>(image_shape[1]) + half_h);
    REQUIRE(y1 < static_cast<float>(image_shape[0]) + half_w);
    REQUIRE(y2 < static_cast<float>(image_shape[0]) + half_w);
  }
}
