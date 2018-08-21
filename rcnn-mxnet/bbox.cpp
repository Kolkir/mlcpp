#include "bbox.h"

Eigen::MatrixXf bbox_overlaps(const Eigen::MatrixXf& boxes,
                              const Eigen::MatrixXf& query_boxes) {
  auto ns = boxes.rows();
  auto ks = query_boxes.rows();
  Eigen::MatrixXf overlaps = Eigen::MatrixXf::Zero(ns, ks);

  Eigen::ArrayXf query_box_area =
      ((query_boxes.block(0, 2, ks, 1) - query_boxes.block(0, 0, ks, 1))
           .array() +
       1) *
      ((query_boxes.block(0, 3, ks, 1) - query_boxes.block(0, 1, ks, 1))
           .array() +
       1);

  Eigen::ArrayXf box_area =
      ((boxes.block(0, 2, ns, 1) - boxes.block(0, 0, ns, 1)).array() + 1) *
      ((boxes.block(0, 3, ns, 1) - boxes.block(0, 1, ns, 1)).array() + 1);

  // Eigen defaults to storing the entry in column-major
  for (Eigen::Index k = 0; k < ks; ++k) {
    for (Eigen::Index n = 0; n < ns; ++n) {
      auto iw = std::min(boxes(n, 2), query_boxes(k, 2)) -
                std::max(boxes(n, 0), query_boxes(k, 0)) + 1;
      if (iw > 0) {
        auto ih = std::min(boxes(n, 3), query_boxes(k, 3)) -
                  std::max(boxes(n, 1), query_boxes(k, 1)) + 1;
        if (ih > 0) {
          auto all_area = box_area(n) + query_box_area(k) - iw * ih;
          overlaps(n, k) = iw * ih / all_area;
        }
      }
    }
  }
  return overlaps;
}

Eigen::MatrixXf bbox_transform(const Eigen::MatrixXf& ex_rois,
                               const Eigen::MatrixXf& gt_rois) {
  assert(ex_rois.rows() == gt_rois.rows());

  auto ex_widths = (ex_rois.col(2) - ex_rois.col(0)).array() + 1.0;
  auto ex_heights = (ex_rois.col(3) - ex_rois.col(1)).array() + 1.0;
  auto ex_ctr_x = ex_rois.col(0).array() + 0.5f * (ex_widths - 1);
  auto ex_ctr_y = ex_rois.col(1).array() + 0.5f * (ex_heights - 1);

  auto gt_widths = (gt_rois.col(2) - gt_rois.col(0)).array() + 1.0;
  auto gt_heights = (gt_rois.col(3) - gt_rois.col(1)).array() + 1.0;
  auto gt_ctr_x = gt_rois.col(0).array() + 0.5f * (gt_widths - 1);
  auto gt_ctr_y = gt_rois.col(1).array() + 0.5f * (gt_heights - 1);

  auto targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-14);
  auto targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-14);
  auto targets_dw = Eigen::log(gt_widths / ex_widths);
  auto targets_dh = Eigen::log(gt_heights / ex_heights);

  Eigen::MatrixXf targets;
  targets << targets_dx, targets_dy, targets_dw, targets_dh;

  return targets;
}
