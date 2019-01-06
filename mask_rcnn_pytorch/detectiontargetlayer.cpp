#include "detectiontargetlayer.h"

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> DetectionTargetLayer(
    const Config& config,
    at::Tensor proposals,
    at::Tensor gt_class_ids,
    at::Tensor gt_boxes,
    at::Tensor gt_masks) {}
