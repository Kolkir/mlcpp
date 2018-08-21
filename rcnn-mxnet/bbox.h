#ifndef BBOX_H
#define BBOX_H

#include <Eigen/Dense>

/*
 * boxes: n * 4 bounding boxes
 * query_boxes: k * 4 bounding boxes
 * return: overlaps: n * k overlaps
 */
Eigen::MatrixXf bbox_overlaps(const Eigen::MatrixXf& boxes,
                              const Eigen::MatrixXf& query_boxes);

/*
 * compute bounding box regression targets from ex_rois to gt_rois
 * ex_rois: [N, 4]
 * gt_rois: [N, 4]
 * return: [N, 4]
 */
Eigen::MatrixXf bbox_transform(const Eigen::MatrixXf& ex_rois,
                               const Eigen::MatrixXf& gt_rois);
#endif  // BBOX_H
