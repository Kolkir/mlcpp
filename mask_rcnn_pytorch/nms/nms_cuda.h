#include <torch/torch.h>

int gpu_nms(at::Tensor keep_out, at::Tensor num_out, at::Tensor boxes, float nms_overlap_thresh);
