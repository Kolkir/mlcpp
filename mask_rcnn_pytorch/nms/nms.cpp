#include "nms.h"
#include <math.h>
#include <torch/torch.h>

int cpu_nms(at::Tensor keep_out,
            at::Tensor num_out,
            at::Tensor boxes,
            at::Tensor order,
            at::Tensor areas,
            float nms_overlap_thresh) {
  // boxes has to be sorted

  // Number of ROIs
  int64_t boxes_num = boxes.size(0);
  int64_t boxes_dim = boxes.size(1);

  int64_t* keep_out_flat = keep_out.contiguous().data<int64_t>();
  float* boxes_flat = boxes.contiguous().data<float>();
  int64_t* order_flat = order.contiguous().data<int64_t>();
  float* areas_flat = areas.contiguous().data<float>();

  at::Tensor suppressed = at::zeros({boxes_num}, at::CPU(at::kByte));
  unsigned char* suppressed_flat = suppressed.data<unsigned char>();

  // nominal indices
  int i, j;
  // sorted indices
  int _i, _j;
  // temp variables for box i's (the box currently under consideration)
  float ix1, iy1, ix2, iy2, iarea;
  // variables for computing overlap with box j (lower scoring box)
  float xx1, yy1, xx2, yy2;
  float w, h;
  float inter, ovr;

  int64_t num_to_keep = 0;
  for (_i = 0; _i < boxes_num; ++_i) {
    i = order_flat[_i];
    if (suppressed_flat[i] == 1) {
      continue;
    }
    keep_out_flat[num_to_keep++] = i;
    ix1 = boxes_flat[i * boxes_dim];
    iy1 = boxes_flat[i * boxes_dim + 1];
    ix2 = boxes_flat[i * boxes_dim + 2];
    iy2 = boxes_flat[i * boxes_dim + 3];
    iarea = areas_flat[i];
    for (_j = _i + 1; _j < boxes_num; ++_j) {
      j = order_flat[_j];
      if (suppressed_flat[j] == 1) {
        continue;
      }
      xx1 = fmaxf(ix1, boxes_flat[j * boxes_dim]);
      yy1 = fmaxf(iy1, boxes_flat[j * boxes_dim + 1]);
      xx2 = fminf(ix2, boxes_flat[j * boxes_dim + 2]);
      yy2 = fminf(iy2, boxes_flat[j * boxes_dim + 3]);
      w = fmaxf(0.0, xx2 - xx1 + 1);
      h = fmaxf(0.0, yy2 - yy1 + 1);
      inter = w * h;
      ovr = inter / (iarea + areas_flat[j] - inter);
      if (ovr >= nms_overlap_thresh) {
        suppressed_flat[j] = 1;
      }
    }
  }

  int64_t* num_out_flat = num_out.contiguous().data<int64_t>();
  *num_out_flat = num_to_keep;
  return 1;
}
