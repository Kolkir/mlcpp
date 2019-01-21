#include <torch/torch.h>

int cpu_nms(at::Tensor keep_out, at::Tensor num_out, at::Tensor boxes, at::Tensor order, at::Tensor areas, float nms_overlap_thresh);