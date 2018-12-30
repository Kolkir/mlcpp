#include <torch/torch.h>

void crop_and_resize_gpu_forward(
    at::Tensor image,
    at::Tensor boxes,      // [y1, x1, y2, x2]
    at::Tensor box_index,  // range in [0, batch_size)
    const float extrapolation_value,
    const int crop_height,
    const int crop_width,
    at::Tensor crops);

void crop_and_resize_gpu_backward(
    at::Tensor grads,
    at::Tensor boxes,       // [y1, x1, y2, x2]
    at::Tensor box_index,   // range in [0, batch_size)
    at::Tensor grads_image  // resize to [bsize, c, hc, wc]
);
