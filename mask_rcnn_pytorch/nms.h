#ifndef NMS_H
#define NMS_H

#include <torch/torch.h>

at::Tensor Nms(at::Tensor dets, float thresh);

#endif  // NMS_H
