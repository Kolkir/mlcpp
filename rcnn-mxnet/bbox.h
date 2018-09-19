#ifndef BBOX_H
#define BBOX_H

#include "params.h"

#include <mxnet-cpp/MxNetCpp.h>
#include <Eigen/Dense>

#include <unordered_set>

using Indices = Eigen::Array<Eigen::Index, Eigen::Dynamic, 1>;

std::pair<Indices, Indices> argmax(const Eigen::MatrixXf& m);

struct WhereVisitor {
  WhereVisitor(bool row = true) : is_row_(row) {}

  void init(const bool& value, Eigen::Index i, Eigen::Index j) {
    operator()(value, i, j);
  }
  void operator()(const bool& value, Eigen::Index i, Eigen::Index j) {
    if (value && data) {
      data->push_back(is_row_ ? i : j);
    }
  }
  bool is_row_{true};
  std::vector<Eigen::Index>* data{nullptr};
};

template <typename T>
std::vector<Eigen::Index> expr_row_indices(const T& expr) {
  WhereVisitor indexes_visitor;
  std::vector<Eigen::Index> result;
  indexes_visitor.data = &result;
  expr.visit(indexes_visitor);
  return result;
}

template <class T, class Rnd>
T random_choice(const T& m, size_t num, Rnd& rnd) {
  std::uniform_int_distribution<size_t> dist(0, m.size() - 1);
  T result;
  result.reserve(num);
  std::unordered_set<size_t> set;

  while (set.size() < num) {
    auto index = dist(rnd);
    if (set.find(index) == set.end()) {
      result.push_back(m[index]);
      set.insert(index);
    }
  }
  return result;
}

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
                               const Eigen::MatrixXf& gt_rois,
                               const std::vector<float>& box_stds = {1, 1, 2,
                                                                     2});

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

/*
 * Generate random sample of ROIs comprising foreground and background examples
 * rois: [n, 4] (x1, y1, x2, y2)
 * gt_boxes: [n, 5] (x1, y1, x2, y2, cls)
 * num_classes: number of classes
 * rois_per_image: total roi number
 * fg_rois_per_image: foreground roi number
 * fg_overlap: overlap threshold for fg rois
 * box_stds: std var of bbox reg
 * return: (labels, rois, bbox_targets, bbox_weights)
 */
std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf>
SampleRois(const Eigen::MatrixXf& rois,
           const Eigen::MatrixXf& gt_boxes,
           int num_classes,
           int rois_per_image,
           int fg_rois_per_image,
           float fg_overlap,
           const std::vector<float>& box_stds);

#endif  // BBOX_H
