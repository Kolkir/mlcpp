#include "crop_and_resize_gpu.h"
#include <torch/torch.h>
#include "cuda/crop_and_resize_kernel.h"

void crop_and_resize_gpu_forward(
    at::Tensor image,
    at::Tensor boxes,      // [y1, x1, y2, x2]
    at::Tensor box_index,  // range in [0, batch_size)
    const float extrapolation_value,
    const uint32_t crop_height,
    const uint32_t crop_width,
    at::Tensor crops) {
  const auto batch_size = image.size(0);
  const auto depth = image.size(1);
  const auto image_height = image.size(2);
  const auto image_width = image.size(3);

  const auto num_boxes = boxes.size(0);

  // init output space
  crops.resize_({num_boxes, depth, crop_height, crop_width});
  crops.zero_();

  CropAndResizeLaucher(image.data<float>(), boxes.data<float>(),
                       box_index.data<int64_t>(), num_boxes, batch_size,
                       image_height, image_width, crop_height, crop_width,
                       depth, extrapolation_value, crops.data<float>());
}

void crop_and_resize_gpu_backward(
    at::Tensor grads,
    at::Tensor boxes,       // [y1, x1, y2, x2]
    at::Tensor box_index,   // range in [0, batch_size)
    at::Tensor grads_image  // resize to [bsize, c, hc, wc]
) {
  // shape
  const auto batch_size = grads_image.size(0);
  const auto depth = grads_image.size(1);
  const auto image_height = grads_image.size(2);
  const auto image_width = grads_image.size(3);

  const auto num_boxes = grads.size(0);
  const auto crop_height = grads.size(2);
  const auto crop_width = grads.size(3);

  // init output space
  grads_image.zero_();

  CropAndResizeBackpropImageLaucher(
      grads.data<float>(), boxes.data<float>(), box_index.data<int64_t>(),
      num_boxes, batch_size, image_height, image_width, crop_height, crop_width,
      depth, grads_image.data<float>());
}
