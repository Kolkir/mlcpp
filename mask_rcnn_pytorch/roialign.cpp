#include "roialign.h"
#include "debug.h"
#include "roialign/crop_and_resize.h"
#include "roialign/crop_and_resize_gpu.h"

at::Tensor PyramidRoiAlign(std::vector<at::Tensor> input,
                           uint32_t pool_size,
                           const std::vector<int32_t>& image_shape) {
  // Currently only supports batchsize 1
  for (auto& i : input)
    i = i.squeeze(0);

  // Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
  auto boxes = input[0];

  // Feature Maps. List of feature maps from different level of the
  // feature pyramid. Each is [batch, height, width, channels]
  std::vector<at::Tensor> feature_maps(std::next(input.begin()), input.end());

  // Assign each ROI to a level in the pyramid based on the ROI area.
  auto boxes_chunk = boxes.chunk(4, /*dim*/ 1);
  auto y1 = boxes_chunk[0];
  auto x1 = boxes_chunk[1];
  auto y2 = boxes_chunk[2];
  auto x2 = boxes_chunk[3];
  auto h = y2 - y1;
  auto w = x2 - x1;

  // Equation 1 in the Feature Pyramid Networks paper. Account for
  // the fact that our coordinates are normalized here.
  // e.g. a 224x224 ROI (in pixels) maps to P4
  auto image_area =
      torch::full({1}, static_cast<float>(image_shape[0] * image_shape[1]));
  if (boxes.is_cuda())
    image_area = image_area.cuda();
  auto roi_level =
      4 + log2(torch::sqrt(h * w) / (224.0 / torch::sqrt(image_area)));
  roi_level = roi_level.round().toType(torch::ScalarType::Int);
  roi_level = roi_level.clamp(2, 5);

  // Loop through levels and apply ROI pooling to each. P2 to P5.
  std::vector<at::Tensor> pooled;
  std::vector<at::Tensor> box_to_level;
  for (uint32_t i = 0, level = 2; level < 6; ++i, ++level) {
    auto ix = roi_level == static_cast<int32_t>(level);
    if (!ix.any().is_nonzero())
      continue;

    ix = ix.nonzero().narrow(1, 0, 1);
    auto level_boxes = boxes.index_select(0, ix.flatten());

    // Keep track of which box is mapped to which level
    box_to_level.push_back(ix.flatten());

    // Stop gradient propogation to ROI proposals
    level_boxes = level_boxes.detach();

    // Crop and Resize
    // From Mask R-CNN paper: "We sample four regular locations, so
    // that we can evaluate either max or average pooling. In fact,
    // interpolating only a single value at each bin center (without
    // pooling) is nearly as effective."
    //
    // Here we use the simplified approach of a single value per bin,
    // which is how it's done in tf.crop_and_resize()
    // Result: [batch * num_boxes, pool_height, pool_width, channels]
    auto ind = torch::zeros({level_boxes.size(0)},
                            at::requires_grad(false).dtype(at::kInt));
    if (level_boxes.is_cuda())
      ind = ind.cuda();
    // CropAndResizeFunction needs batch dimension
    feature_maps[i] = feature_maps[i].unsqueeze(0);

    torch::Tensor pooled_features = torch::empty({}, at::dtype(at::kFloat));
    if (level_boxes.is_cuda())
      pooled_features = pooled_features.cuda();

    if (boxes.is_cuda()) {
      crop_and_resize_gpu_forward(feature_maps[i], level_boxes, ind, 0,
                                  pool_size, pool_size, pooled_features);
    } else {
      crop_and_resize_forward(feature_maps[i], level_boxes, ind, 0, pool_size,
                              pool_size, pooled_features);
    }
    pooled.push_back(pooled_features);
  }

  // Pack pooled features into one tensor
  auto pooled_res = torch::cat(at::TensorList(pooled), /*dim*/ 0);

  // Pack box_to_level mapping into one array and add another
  // column representing the order of pooled boxes
  auto box_to_level_res = torch::cat(at::TensorList(box_to_level), /*dim*/ 0);

  // Rearrange pooled features to match the order of the original boxes
  std::tie(std::ignore, box_to_level_res) = torch::sort(box_to_level_res);
  pooled_res = pooled_res.index_select(0, box_to_level_res.flatten());

  return pooled_res;
}
