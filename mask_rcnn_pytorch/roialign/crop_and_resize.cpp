#include "crop_and_resize.h"
#include <torch/torch.h>
#include <cmath>
#include <iostream>

void CropAndResizePerBox(const float* image_data,
                         const int64_t batch_size,
                         const int64_t depth,
                         const int64_t image_height,
                         const int64_t image_width,

                         const float* boxes_data,
                         const int64_t* box_index_data,
                         const int64_t start_box,
                         const int64_t limit_box,

                         float* crops_data,
                         const int64_t crop_height,
                         const int64_t crop_width,
                         const float extrapolation_value) {
  const int64_t image_channel_elements = image_height * image_width;
  const int64_t image_elements = depth * image_channel_elements;

  const int64_t channel_elements = crop_height * crop_width;
  const int64_t crop_elements = depth * channel_elements;

  int64_t b = 0;
#pragma omp parallel for
  for (b = start_box; b < limit_box; ++b) {
    const float* box = boxes_data + b * 4;
    const float y1 = box[0];
    const float x1 = box[1];
    const float y2 = box[2];
    const float x2 = box[3];

    const int64_t b_in = box_index_data[b];
    if (b_in < 0 || b_in >= batch_size) {
      std::cout << "Error: batch_index " << b_in << " out of range [0, "
                << batch_size << ")\n";
      exit(-1);
    }

    const float height_scale =
        (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                          : 0;
    const float width_scale =
        (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;

    for (int y = 0; y < crop_height; ++y) {
      const float in_y = (crop_height > 1)
                             ? y1 * (image_height - 1) + y * height_scale
                             : 0.5f * (y1 + y2) * (image_height - 1);

      if (in_y < 0 || in_y > image_height - 1) {
        for (int x = 0; x < crop_width; ++x) {
          for (int d = 0; d < depth; ++d) {
            // crops(b, y, x, d) = extrapolation_value;
            crops_data[crop_elements * b + channel_elements * d +
                       y * crop_width + x] = extrapolation_value;
          }
        }
        continue;
      }

      const int32_t top_y_index = static_cast<int32_t>(floorf(in_y));
      const int32_t bottom_y_index = static_cast<int32_t>(ceilf(in_y));
      const float y_lerp = in_y - top_y_index;

      for (int x = 0; x < crop_width; ++x) {
        const float in_x = (crop_width > 1)
                               ? x1 * (image_width - 1) + x * width_scale
                               : 0.5f * (x1 + x2) * (image_width - 1);
        if (in_x < 0 || in_x > image_width - 1) {
          for (int d = 0; d < depth; ++d) {
            crops_data[crop_elements * b + channel_elements * d +
                       y * crop_width + x] = extrapolation_value;
          }
          continue;
        }

        const int32_t left_x_index = static_cast<int32_t>(floorf(in_x));
        const int32_t right_x_index = static_cast<int32_t>(ceilf(in_x));
        const float x_lerp = in_x - left_x_index;

        for (int d = 0; d < depth; ++d) {
          const float* pimage =
              image_data + b_in * image_elements + d * image_channel_elements;

          const float top_left =
              pimage[top_y_index * image_width + left_x_index];
          const float top_right =
              pimage[top_y_index * image_width + right_x_index];
          const float bottom_left =
              pimage[bottom_y_index * image_width + left_x_index];
          const float bottom_right =
              pimage[bottom_y_index * image_width + right_x_index];

          const float top = top_left + (top_right - top_left) * x_lerp;
          const float bottom =
              bottom_left + (bottom_right - bottom_left) * x_lerp;

          crops_data[crop_elements * b + channel_elements * d + y * crop_width +
                     x] = top + (bottom - top) * y_lerp;
        }
      }  // end for x
    }    // end for y
  }      // end for b
}

void crop_and_resize_forward(at::Tensor image,
                             at::Tensor boxes,      // [y1, x1, y2, x2]
                             at::Tensor box_index,  // range in [0, batch_size)
                             float extrapolation_value,
                             uint32_t crop_height,
                             uint32_t crop_width,
                             at::Tensor crops) {
  const auto batch_size = image.size(0);
  const auto depth = image.size(1);
  const auto image_height = image.size(2);
  const auto image_width = image.size(3);

  const auto num_boxes = boxes.size(0);

  // init output space
  crops.resize_({num_boxes, depth, crop_height, crop_width});
  crops.zero_();

  // crop_and_resize for each box
  CropAndResizePerBox(
      image.data<float>(), batch_size, depth, image_height, image_width,

      boxes.data<float>(), box_index.data<int64_t>(), 0, num_boxes,

      crops.data<float>(), crop_height, crop_width, extrapolation_value);
}

void crop_and_resize_backward(
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

  // n_elements
  const auto image_channel_elements = image_height * image_width;
  const auto image_elements = depth * image_channel_elements;

  const auto channel_elements = crop_height * crop_width;
  const auto crop_elements = depth * channel_elements;

  // init output space
  grads_image.zero_();

  // data pointer
  const float* grads_data = grads.data<float>();
  const float* boxes_data = boxes.data<float>();
  const int* box_index_data = box_index.data<int>();
  float* grads_image_data = grads_image.data<float>();

  for (int b = 0; b < num_boxes; ++b) {
    const float* box = boxes_data + b * 4;
    const float y1 = box[0];
    const float x1 = box[1];
    const float y2 = box[2];
    const float x2 = box[3];

    const int b_in = box_index_data[b];
    if (b_in < 0 || b_in >= batch_size) {
      std::cout << "Error: batch_index " << b_in << " out of range [0, "
                << batch_size << ")\n";
      exit(-1);
    }

    const float height_scale =
        (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
                          : 0;
    const float width_scale =
        (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;

    for (int y = 0; y < crop_height; ++y) {
      const float in_y = (crop_height > 1)
                             ? y1 * (image_height - 1) + y * height_scale
                             : 0.5f * (y1 + y2) * (image_height - 1);
      if (in_y < 0 || in_y > image_height - 1) {
        continue;
      }
      const int32_t top_y_index = static_cast<int32_t>(floorf(in_y));
      const int32_t bottom_y_index = static_cast<int32_t>(ceilf(in_y));
      const float y_lerp = in_y - top_y_index;

      for (int x = 0; x < crop_width; ++x) {
        const float in_x = (crop_width > 1)
                               ? x1 * (image_width - 1) + x * width_scale
                               : 0.5f * (x1 + x2) * (image_width - 1);
        if (in_x < 0 || in_x > image_width - 1) {
          continue;
        }
        const int32_t left_x_index = static_cast<int32_t>(floorf(in_x));
        const int32_t right_x_index = static_cast<int32_t>(ceilf(in_x));
        const float x_lerp = in_x - left_x_index;

        for (int d = 0; d < depth; ++d) {
          float* pimage = grads_image_data + b_in * image_elements +
                          d * image_channel_elements;
          const float grad_val =
              grads_data[crop_elements * b + channel_elements * d +
                         y * crop_width + x];

          const float dtop = (1 - y_lerp) * grad_val;
          pimage[top_y_index * image_width + left_x_index] +=
              (1 - x_lerp) * dtop;
          pimage[top_y_index * image_width + right_x_index] += x_lerp * dtop;

          const float dbottom = y_lerp * grad_val;
          pimage[bottom_y_index * image_width + left_x_index] +=
              (1 - x_lerp) * dbottom;
          pimage[bottom_y_index * image_width + right_x_index] +=
              x_lerp * dbottom;
        }  // end d
      }    // end x
    }      // end y
  }        // end b
}
