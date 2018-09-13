#include "anchorsampler.h"
#include "bbox.h"

#include <iostream>

AnchorSampler::AnchorSampler(const Params& params)
    : allowed_border_(params.rpn_allowed_border),
      num_batch_(params.rpn_batch_rois),
      fg_fraction_(params.rpn_fg_fraction),
      fg_overlap_(params.rpn_fg_overlap),
      bg_overlap_(params.rpn_bg_overlap) {
  num_fg_ = static_cast<Eigen::Index>(num_batch_ * fg_fraction_);
}

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf>
AnchorSampler::Assign(const Eigen::MatrixXf& anchors,
                      const Eigen::MatrixXf& gt_boxes,
                      float im_width,
                      float im_height) {
  auto n_anchors = anchors.rows();
  std::cout << "Image width " << im_width << "Image height " << im_height
            << std::endl;
  std::cout << "All anchors count " << n_anchors << std::endl;

  // filter out anchors outside the image region
  Eigen::ArrayXf indices_base = Eigen::ArrayXf::Constant(n_anchors, 1.f);

  Eigen::ArrayXf inds_inside =
      (anchors.block(0, 0, n_anchors, 1).array() >= -allowed_border_)
          .select(indices_base, -indices_base);
  inds_inside = (anchors.block(0, 1, n_anchors, 1).array() >= -allowed_border_)
                    .select(inds_inside, -indices_base);
  inds_inside =
      (anchors.block(0, 2, n_anchors, 1).array() < im_width + allowed_border_)
          .select(inds_inside, -indices_base);
  inds_inside =
      (anchors.block(0, 3, n_anchors, 1).array() < im_height + allowed_border_)
          .select(inds_inside, -indices_base);

  auto num_valid = (inds_inside > 0).count();

  Eigen::MatrixXf valid_anchors(num_valid, anchors.cols());
  for (Eigen::Index i = 0, j = 0; i < n_anchors; ++i) {
    if (inds_inside[i] >= 1.f) {
      valid_anchors.row(j++) = anchors.row(i);
    }
  }
  std::cout << "Valid anchors count " << num_valid << std::endl;

  // label: 1 is positive, 0 is negative, -1 is dont care
  Eigen::MatrixXf labels = Eigen::MatrixXf::Constant(num_valid, 1, -1.f);
  Eigen::MatrixXf bbox_targets = Eigen::MatrixXf::Zero(num_valid, 4);
  Eigen::MatrixXf bbox_weights = Eigen::MatrixXf::Zero(num_valid, 4);

  // sample for positive labels
  if (gt_boxes.rows() > 0) {
    // overlap between the anchors and the gt boxes
    // result is 2d matrix with ratio of overlap (anchors, gt_boxes)
    auto overlaps = bbox_overlaps(valid_anchors, gt_boxes);

    // fg anchors: anchor with highest overlap for each gt
    auto gt_max_overlaps =
        (overlaps.array() ==
         overlaps.colwise().maxCoeff().array().replicate(overlaps.rows(), 1))
            .matrix()
            .rowwise()
            .any();
    labels = (gt_max_overlaps.array() > 0).select(1.f, labels);

    // fg anchors: anchor with overlap > iou thresh
    auto max_overlaps = overlaps.rowwise().maxCoeff();
    labels = (max_overlaps.array() >= fg_overlap_).select(1.f, labels);

    // bg anchors: anchor with overlap < iou thresh
    labels = (max_overlaps.array() < bg_overlap_).select(0.f, labels);

    // subsample positive anchors
    Eigen::Index fg_labels_count = (labels.array() >= 1.f).count();
    if (fg_labels_count > num_fg_) {
      auto disable_inds =
          random_choice(0, fg_labels_count - 1, fg_labels_count - num_fg_);
      labels = (disable_inds.array() > 0).select(-1.f, labels);
    }

    // subsample negative anchors
    Eigen::Index bg_labels_count = (labels.array() == 0.f).count();
    auto max_neg = num_batch_ - std::min(num_fg_, fg_labels_count);
    if (bg_labels_count > max_neg) {
      auto disable_inds =
          random_choice(0, bg_labels_count - 1, bg_labels_count - max_neg);
      labels = (disable_inds.array() > 0).select(-1.f, labels);
    }

    // make gt_box correspondence for each positive anchor
    Eigen::MatrixXf anchors_gt_boxes(valid_anchors.rows(), gt_boxes.cols());
    auto anchor_box_inds = argmax(overlaps).second;
    assert(labels.rows() == anchor_box_inds.rows());
    for (Eigen::Index i = 0, j = 0; i < num_valid; ++i) {
      anchors_gt_boxes.row(j++) = gt_boxes.row(anchor_box_inds(i));
    }

    // calculate anchor vs bbox offsets
    auto raw_bbox_targets = bbox_transform(valid_anchors, anchors_gt_boxes);
    bbox_targets = (labels.replicate(1, bbox_targets.cols()).array() >= 1.f)
                       .select(1.f, raw_bbox_targets);

    // only fg anchors has bbox_targets
    bbox_weights = (labels.replicate(1, bbox_weights.cols()).array() >= 1.f)
                       .select(1.f, bbox_weights);
  } else {
    // randomly draw bg anchors
    auto bg_inds = random_choice(0, labels.rows() - 1, num_batch_);
    labels = (bg_inds.array() > 0).select(0, labels);
  }

  // make final result
  Eigen::MatrixXf all_labels = Eigen::MatrixXf::Constant(n_anchors, 1, -1.f);
  Eigen::MatrixXf all_bbox_targets = Eigen::MatrixXf::Zero(n_anchors, 4);
  Eigen::MatrixXf all_bbox_weights = Eigen::MatrixXf::Zero(n_anchors, 4);
  for (Eigen::Index i = 0, j = 0; i < n_anchors; ++i) {
    if (inds_inside[i] >= 1.f) {
      all_labels(i, 0) = labels(j, 0);
      all_bbox_targets.row(i) = bbox_targets.row(j);
      all_bbox_weights.row(i) = bbox_weights.row(j);
      ++j;
    }
  }

  return {all_labels, all_bbox_targets, all_bbox_weights};
}
