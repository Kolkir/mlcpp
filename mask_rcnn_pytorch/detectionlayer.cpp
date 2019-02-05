#include "detectionlayer.h"
#include "boxutils.h"
#include "debug.h"
#include "nms.h"
#include "nnutils.h"
#include "proposallayer.h"

namespace {

/*
 * Refine classified proposals and filter overlaps and return final detections.
 * Inputs:
 *     rois: [N, (y1, x1, y2, x2)] in normalized coordinates
 *     probs: [N, num_classes]. Class probabilities.
 *     deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
 *             bounding box deltas.
 *     window: (y1, x1, y2, x2) in image coordinates. The part
 *             of the image that contains the image excluding the padding.
 * Returns
 *       detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
 */
at::Tensor RefineDetections(at::Tensor rois,
                            at::Tensor probs,
                            at::Tensor deltas,
                            const Window& window,
                            const Config& config) {
  // Class IDs per ROI
  at::Tensor class_ids;
  std::tie(std::ignore, class_ids) = torch::max(probs, /*dim*/ 1);

  // Class probability of the top class of each ROI
  // Class-specific bounding box deltas
  auto idx = torch::arange(class_ids.size(0), at::dtype(at::kLong));
  if (config.gpu_count > 0)
    idx = idx.cuda();

  auto class_scores = probs.index({idx, class_ids});
  auto deltas_specific = deltas.index({idx, class_ids});

  // Apply bounding box deltas
  // Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
  auto std_dev = torch::tensor(config.rpn_bbox_std_dev,
                               at::dtype(at::kFloat).requires_grad(false));
  if (config.gpu_count > 0)
    std_dev = std_dev.cuda();
  auto refined_rois = ApplyBoxDeltas(rois, deltas_specific * std_dev);

  // Convert coordiates to image domain
  auto height = static_cast<float>(config.image_shape[0]);
  auto width = static_cast<float>(config.image_shape[1]);

  auto scale = torch::tensor({height, width, height, width},
                             at::dtype(at::kFloat).requires_grad(false));
  if (config.gpu_count > 0)
    scale = scale.cuda();
  refined_rois *= scale;

  // Clip boxes to image window
  refined_rois = ClipToWindow(window, refined_rois);

  // Round and cast to  int since we're deadling with pixels now
  refined_rois = torch::round(refined_rois);

  // TODO: (Legacy)Filter out  boxes with zero area

  // Filter out background boxes
  auto keep_bool = class_ids > 0;

  // Filter out low  confidence boxes
  if (config.detection_min_confidence > 0) {
    keep_bool = keep_bool * (class_scores >= config.detection_min_confidence);
  }

  auto keep = torch::nonzero(keep_bool)
                  .narrow(1, 0, 1)
                  .to(at::dtype(at::kLong))
                  .squeeze();

  // Apply per-class NMS
  auto pre_nms_class_ids = class_ids.take(keep);
  auto pre_nms_scores = class_scores.take(keep);
  auto pre_nms_rois = refined_rois.index_select(0, keep);

  auto nms_class_ids = unique1d(pre_nms_class_ids);
  at::Tensor nms_keep =
      torch::empty({0}, at::dtype(at::kLong).requires_grad(false));
  if (config.gpu_count > 0)
    nms_keep = nms_keep.cuda();
  for (int64_t i = 0; i < nms_class_ids.size(0); ++i) {
    auto class_id = nms_class_ids[i];
    //    // Pick detections of this class
    auto ixs =
        torch::nonzero(pre_nms_class_ids == class_id).narrow(1, 0, 1).squeeze();

    // Sort
    auto ix_rois = pre_nms_rois.index_select(0, ixs);

    auto ix_scores = pre_nms_scores.index_select(0, ixs);
    at::Tensor order;
    std::tie(ix_scores, order) =
        ix_scores.unsqueeze(1).sort(0, /*descending*/ true);
    order = order.squeeze();
    ix_rois = ix_rois.index_select(0, order);

    auto class_keep = Nms(torch::cat({ix_rois, ix_scores}, /*dim*/ 1),
                          config.detection_nms_threshold);

    // Map indicies
    class_keep = keep.take(ixs.take(order.take(class_keep)));

    if (i == 0)
      nms_keep = class_keep;
    else
      nms_keep = unique1d(torch::cat({nms_keep, class_keep}));
  }
  keep = intersect1d(keep, nms_keep);

  // Keep top detections
  auto scores = class_scores.take(keep).unsqueeze(1);
  at::Tensor top_ids;
  std::tie(scores, top_ids) = scores.sort(0, /*descending*/ true);
  auto roi_count = config.detection_max_instances;
  top_ids = top_ids.narrow(0, 0, std::min(roi_count, top_ids.size(0)));
  keep = keep.take(top_ids.squeeze());

  // Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
  // Coordinates are in image domain.
  auto result = torch::cat(
      {refined_rois.index_select(0, keep),
       class_ids.index_select(0, keep).unsqueeze(1).to(at::dtype(at::kFloat)),
       class_scores.index_select(0, keep).unsqueeze(1)},
      /*dim*/ 1);
  return result;
}
}  // namespace
at::Tensor DetectionLayer(const Config& config,
                          at::Tensor rois,
                          at::Tensor mrcnn_class,
                          at::Tensor mrcnn_bbox,
                          const std::vector<ImageMeta>& image_meta) {
  // Currently only supports batchsize 1
  rois = rois.squeeze(0);
  auto detections = RefineDetections(rois, mrcnn_class, mrcnn_bbox,
                                     image_meta[0].window, config);
  return detections;
}
