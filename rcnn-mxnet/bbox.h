#ifndef BBOX_H
#define BBOX_H

#include "params.h"

#include <mxnet-cpp/MxNetCpp.h>
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

/*
 * Transform the set of class-agnostic boxes into class-specific boxes
 * by applying the predicted offsets (box_deltas)
 * boxes: !important [N 4]
 * box_deltas: [N, 4 * num_classes]
 * return: [N 4 * num_classes]
 */
Eigen::MatrixXf bbox_pred(const Eigen::MatrixXf& boxes,
                          const Eigen::MatrixXf& box_deltas,
                          const Eigen::MatrixXf& box_stds);

/*
 * Clip boxes to image boundaries.
 * boxes: [N, 4* num_classes]
 * im_shape: tuple of 2
 * return: [N, 4* num_classes] x1,y1,x2,y2
 */
Eigen::MatrixXf clip_boxes(const Eigen::MatrixXf& boxes,
                           float width,
                           float height);

Eigen::MatrixXf NDArray2ToEigen(const mxnet::cpp::NDArray& value);
Eigen::MatrixXf NDArray3ToEigen(
    const mxnet::cpp::NDArray& value);  // ignore first dimension

struct Detection {
  long class_id = -1;
  float x1 = 0;
  float y1 = 0;
  float x2 = 0;
  float y2 = 0;
  float score = 0;
  float area() { return (x2 - x1 + 1) * (y2 - y1 + 1); }
};

std::vector<Detection> DecodePredictions(const Eigen::MatrixXf& rois,
                                         const Eigen::MatrixXf& scores,
                                         const Eigen::MatrixXf& bbox_deltas,
                                         const Eigen::MatrixXf& im_info,
                                         const Params& params);

/*
 * greedily select boxes with high confidence and overlap with current maximum
 * <= thresh rule out overlap >= thresh
 */
void nms(std::vector<Detection>& predictions, float nms_thresh);

#endif  // BBOX_H
