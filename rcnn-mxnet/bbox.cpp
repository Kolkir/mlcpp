#include "bbox.h"

#include <random>
#include <type_traits>

namespace {
struct ArgmaxVisitor {
  ArgmaxVisitor(Eigen::Index rows, Eigen::Index cols)
      : max_rows(cols, 1), max_cols(rows, 1) {}
  void init(const bool& value, Eigen::Index i, Eigen::Index j) {
    operator()(value, i, j);
  }
  void operator()(const bool& value, Eigen::Index i, Eigen::Index j) {
    if (value) {
      max_cols(i, 0) = j;
      max_rows(j, 0) = i;
    }
  }
  Indices max_rows;
  Indices max_cols;
};
}  // namespace

std::pair<Indices, Indices> argmax(const Eigen::MatrixXf& m) {
  auto max_pos =
      (m.array() == m.rowwise().maxCoeff().array().replicate(1, m.cols()))
          .matrix();
  ArgmaxVisitor visitor(m.rows(), m.cols());
  max_pos.visit(visitor);
  return {visitor.max_rows, visitor.max_cols};
}

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
                               const Eigen::MatrixXf& gt_rois,
                               const std::vector<float>& box_stds) {
  assert(ex_rois.rows() == gt_rois.rows());

  auto ex_widths = (ex_rois.col(2) - ex_rois.col(0)).array() + 1.0;
  auto ex_heights = (ex_rois.col(3) - ex_rois.col(1)).array() + 1.0;
  auto ex_ctr_x = ex_rois.col(0).array() + 0.5f * (ex_widths - 1);
  auto ex_ctr_y = ex_rois.col(1).array() + 0.5f * (ex_heights - 1);

  auto gt_widths = (gt_rois.col(2) - gt_rois.col(0)).array() + 1.0;
  auto gt_heights = (gt_rois.col(3) - gt_rois.col(1)).array() + 1.0;
  auto gt_ctr_x = gt_rois.col(0).array() + 0.5f * (gt_widths - 1);
  auto gt_ctr_y = gt_rois.col(1).array() + 0.5f * (gt_heights - 1);

  auto targets_dx =
      ((gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-14)).array() / box_stds[0];
  auto targets_dy =
      ((gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-14)).array() / box_stds[1];
  auto targets_dw = Eigen::log(gt_widths / ex_widths).array() / box_stds[2];
  auto targets_dh = Eigen::log(gt_heights / ex_heights).array() / box_stds[3];

  Eigen::MatrixXf targets(ex_rois.rows(), ex_rois.cols());
  targets << targets_dx, targets_dy, targets_dw, targets_dh;

  return targets;
}

std::vector<Detection> DecodePredictions(const Eigen::MatrixXf& rois,
                                         const Eigen::MatrixXf& scores,
                                         const Eigen::MatrixXf& bbox_deltas,
                                         const Eigen::MatrixXf& im_info,
                                         const Params& params) {
  auto height = im_info(0, 0);
  auto width = im_info(0, 1);
  auto scale = im_info(0, 2);

  // post processing
  auto pred_boxes =
      bbox_pred(rois.block(0, 1, rois.rows(), rois.cols() - 1), bbox_deltas,
                Eigen::Map<const Eigen::MatrixXf>(
                    params.rcnn_bbox_stds.data(), 1,
                    static_cast<Eigen::Index>(params.rcnn_bbox_stds.size())));
  pred_boxes = clip_boxes(pred_boxes, width, height);

  // we used scaled image & roi to train, so it is necessary to transform them
  // back
  pred_boxes = pred_boxes.array() / scale;

  std::vector<Detection> result;
  std::vector<Detection> one_class_result;
  one_class_result.reserve(static_cast<size_t>(scores.rows()));
  // convert to per class detection results
  for (Eigen::Index c = 1; c < scores.cols(); ++c) {
    Eigen::ArrayXf filtered_scores =
        (scores.col(c).array() > params.rcnn_conf_thresh)
            .select(scores.col(c), -1)
            .array();
    one_class_result.clear();
    for (Eigen::Index i = 0; i < filtered_scores.size(); ++i) {
      if (filtered_scores(i) >= 0) {
        one_class_result.emplace_back();
        one_class_result.back().class_id = c;
        one_class_result.back().score = filtered_scores(i);
        one_class_result.back().x1 = pred_boxes(i, 0);
        one_class_result.back().y1 = pred_boxes(i, 1);
        one_class_result.back().x2 = pred_boxes(i, 2);
        one_class_result.back().y2 = pred_boxes(i, 3);
      }
    }
    nms(one_class_result, params.rpn_nms_thresh);
    result.insert(result.end(), one_class_result.begin(),
                  one_class_result.end());
  }
  return result;
}

void nms(std::vector<Detection>& predictions, float nms_thresh) {
  using I = std::vector<Detection>::iterator;
  std::vector<I> inds(predictions.size());
  std::iota(inds.begin(), inds.end(), predictions.begin());
  std::sort(inds.begin(), inds.end(),
            [](I a, I b) { return a->score > b->score; });
  std::vector<Detection> keep;
  while (!inds.empty()) {
    auto i = inds.front();
    keep.push_back(*i);

    auto suppress = std::remove_if(inds.begin(), inds.end(), [&](I j) {
      if (i == j)
        return true;

      // find the largest (x, y) coordinates for the start of the bounding box
      // and the smallest (x, y) coordinates for the end of the bounding box
      auto xx1 = std::max(i->x1, j->x1);
      auto yy1 = std::max(i->y1, j->y1);
      auto xx2 = std::min(i->x2, j->x2);
      auto yy2 = std::min(i->y2, j->y2);

      // compute the width and height of the bounding box
      auto w = std::max(0.f, xx2 - xx1 + 1.f);
      auto h = std::max(0.f, yy2 - yy1 + 1.f);

      // compute the ratio of overlap between the computed bounding box and the
      // bounding box in the area list
      auto inter = w * h;
      auto overlap = inter / (i->area() + j->area() - inter);

      // if there is sufficient overlap, suppress the current bounding box
      if (overlap > nms_thresh)
        return true;
      else
        return false;
    });

    inds.erase(suppress, inds.end());
  }
  predictions = keep;
}

template <typename T>
auto SliceColumns(T& val, Eigen::Index start, Eigen::Index stride) {
  auto cols = static_cast<Eigen::Index>(
      std::ceil(val.cols() / static_cast<double>(stride)));
  using Maptype =
      typename std::conditional<std::is_const<T>::value, const Eigen::MatrixXf,
                                Eigen::MatrixXf>::type;

  return Eigen::Map<Maptype, 0, Eigen::OuterStride<>>(
      val.rightCols(val.cols() - start).data(), val.rows(), cols,
      Eigen::OuterStride<>(val.outerStride() * stride));
}

Eigen::MatrixXf bbox_pred(const Eigen::MatrixXf& boxes,
                          const Eigen::MatrixXf& box_deltas,
                          const Eigen::MatrixXf& box_stds) {
  using namespace Eigen;
  if (boxes.rows() == 0)
    return Eigen::MatrixXf::Zero(0, box_deltas.cols());

  ArrayXf widths = (boxes.col(2) - boxes.col(0)).array() + 1.0f;
  ArrayXf heights = (boxes.col(3) - boxes.col(1)).array() + 1.0f;
  ArrayXf ctr_x = boxes.col(0).array() + 0.5f * (widths - 1.0f);
  ArrayXf ctr_y = boxes.col(1).array() + 0.5f * (heights - 1.0f);

  MatrixXf dx = SliceColumns(box_deltas, 0, 4).array() * box_stds(0, 0);
  MatrixXf dy = SliceColumns(box_deltas, 1, 4).array() * box_stds(0, 1);
  MatrixXf dw = SliceColumns(box_deltas, 2, 4).array() * box_stds(0, 2);
  MatrixXf dh = SliceColumns(box_deltas, 3, 4).array() * box_stds(0, 3);

  MatrixXf pred_ctr_x =
      (dx.array().colwise() * widths).array().colwise() + ctr_x;
  MatrixXf pred_ctr_y =
      (dy.array().colwise() * heights).array().colwise() + ctr_y;
  MatrixXf pred_w = dw.array().exp().colwise() * widths;
  MatrixXf pred_h = dh.array().exp().colwise() * heights;

  MatrixXf pred_boxes = MatrixXf::Zero(box_deltas.rows(), box_deltas.cols());
  // x1
  SliceColumns(pred_boxes, 0, 4) =
      pred_ctr_x.array() - 0.5f * (pred_w.array() - 1.0f);
  // y1
  SliceColumns(pred_boxes, 1, 4) =
      pred_ctr_y.array() - 0.5 * (pred_h.array() - 1.0f);
  // x2
  SliceColumns(pred_boxes, 2, 4) =
      pred_ctr_x.array() + 0.5f * (pred_w.array() - 1.0f);
  // y2
  SliceColumns(pred_boxes, 3, 4) =
      pred_ctr_y.array() + 0.5f * (pred_h.array() - 1.0f);

  return pred_boxes;
}

Eigen::MatrixXf clip_boxes(const Eigen::MatrixXf& boxes,
                           float width,
                           float height) {
  using namespace Eigen;
  MatrixXf boxes_clipped = boxes;
  // x1 >= 0
  MatrixXf bx = SliceColumns(boxes_clipped, 0, 4);
  bx = (bx.array() < width).select(bx, width - 1);
  bx = (bx.array() >= 0).select(bx, 0);
  // y1 >= 0
  MatrixXf by = SliceColumns(boxes_clipped, 1, 4);
  by = (by.array() < height).select(by, height - 1);
  by = (by.array() >= 0).select(by, 0);
  // x2 < width
  MatrixXf bw = SliceColumns(boxes_clipped, 2, 4);
  bw = (bw.array() < height).select(bw, width - 1);
  bw = (bw.array() >= 0).select(bw, 0);
  // y2 < height
  MatrixXf bh = SliceColumns(boxes_clipped, 3, 4);
  bh = (bh.array() < height).select(bh, height - 1);
  bh = (bh.array() >= 0).select(bh, 0);
  return boxes_clipped;
}

Eigen::MatrixXf NDArray2ToEigen(const mxnet::cpp::NDArray& value) {
  assert(value.GetShape().size() == 2);
  //  auto result =
  //      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
  //                                     Eigen::RowMajor>>(
  //          value.GetData(), value.GetShape()[0], value.GetShape()[1]);

  Eigen::MatrixXf result(value.GetShape()[0], value.GetShape()[1]);
  for (Eigen::Index x = 0; x < result.cols(); ++x) {
    for (Eigen::Index y = 0; y < result.rows(); ++y) {
      result(y, x) = value.At(y, x);
    }
  }
  return result;
}

Eigen::MatrixXf NDArray3ToEigen(const mxnet::cpp::NDArray& value) {
  assert(value.GetShape().size() == 3);
  auto result =
      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>(
          value.GetData(), value.GetShape()[1], value.GetShape()[2]);
  return result;
}

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf>
SampleRois(const Eigen::MatrixXf& rois,
           const Eigen::MatrixXf& gt_boxes,
           int num_classes,
           int rois_per_image,
           int fg_rois_per_image,
           float fg_overlap,
           const std::vector<float>& box_stds) {
  auto overlaps = bbox_overlaps(rois, gt_boxes);
  auto gt_assignment = argmax(overlaps).second;

  Eigen::MatrixXf labels(gt_assignment.rows(), 1);
  for (Eigen::Index i = 0; i < gt_assignment.size(); ++i) {
    Eigen::Index j = gt_assignment(i);
    labels(i, 0) = gt_boxes(j, 4);
  }
  Eigen::MatrixXf max_overlaps = overlaps.rowwise().maxCoeff();

  std::vector<Eigen::Index> fg_indexes;
  std::vector<Eigen::Index> bg_indexes;
  WhereVisitor indexes_visitor;

  std::random_device rd;
  std::mt19937 mt(rd());

  // select foreground RoI with FG_THRESH overlap
  auto fg_indexes_expr = (max_overlaps.array() >= fg_overlap);
  indexes_visitor.data = &fg_indexes;
  fg_indexes_expr.visit(indexes_visitor);
  // guard against the case when an image has fewer than fg_rois_per_image
  // foreground RoIs
  auto fg_indexes_count = fg_indexes_expr.count();
  auto fg_rois_this_image =
      std::min(static_cast<Eigen::Index>(fg_rois_per_image), fg_indexes_count);
  // sample foreground regions without replacement
  if (fg_indexes_count > fg_rois_this_image) {
    fg_indexes =
        random_choice(fg_indexes, static_cast<size_t>(fg_rois_this_image), mt);
  }

  // select background RoIs as those within [0, FG_THRESH)
  auto bg_indexes_expr = (max_overlaps.array() < fg_overlap);
  indexes_visitor.data = &bg_indexes;
  bg_indexes_expr.visit(indexes_visitor);
  // compute number of background RoIs to take from this image (guarding
  // against there being fewer than desired)
  auto bg_rois_this_image = rois_per_image - fg_rois_this_image;
  auto bg_indexes_count = bg_indexes_expr.count();
  bg_rois_this_image = std::min(bg_rois_this_image, bg_indexes_count);
  // sample bg rois without replacement
  if (bg_indexes_count > bg_rois_this_image) {
    bg_indexes =
        random_choice(bg_indexes, static_cast<size_t>(bg_rois_this_image), mt);
  }

  // indexes selected
  std::vector<Eigen::Index> keep_indexes;
  keep_indexes.insert(keep_indexes.end(), fg_indexes.begin(), fg_indexes.end());
  keep_indexes.insert(keep_indexes.end(), bg_indexes.begin(), bg_indexes.end());

  // pad more bg rois to ensure a fixed minibatch size
  while (keep_indexes.size() < static_cast<size_t>(rois_per_image)) {
    auto gap = std::min(bg_indexes.size(), static_cast<size_t>(rois_per_image) -
                                               keep_indexes.size());
    auto gap_indexes = random_choice(bg_indexes, gap, mt);
    keep_indexes.insert(keep_indexes.end(), gap_indexes.begin(),
                        gap_indexes.end());
  }

  // sample rois and labels
  assert(static_cast<size_t>(rois_per_image) == keep_indexes.size());
  Eigen::MatrixXf b_rois(rois_per_image, rois.cols());
  Eigen::MatrixXf b_labels(rois_per_image, 1);
  int j = 0;
  for (auto i : keep_indexes) {
    b_rois.row(j) = rois.row(i);
    // set labels of bg rois to be 0
    if (j >= fg_rois_this_image)
      b_labels(j, 0) = 0;
    else
      b_labels(j, 0) = labels(i, 0);
    ++j;
  }

  // load or compute bbox_target
  Eigen::MatrixXf boxes(rois_per_image, 4);
  int b = 0;
  for (auto i : keep_indexes) {
    Eigen::Index j = gt_assignment(i);
    boxes.row(b) = gt_boxes.row(j).leftCols(4);
    ++b;
  }

  auto targets = bbox_transform(b_rois, boxes, box_stds);

  Eigen::MatrixXf bbox_targets =
      Eigen::MatrixXf::Zero(rois_per_image, 4 * num_classes);
  Eigen::MatrixXf bbox_weights =
      Eigen::MatrixXf::Zero(rois_per_image, 4 * num_classes);
  for (Eigen::Index i = 0; i < fg_rois_this_image; ++i) {
    auto cls_ind = static_cast<int>(b_labels(i));
    bbox_targets.row(i).block(0, cls_ind * 4, 1, 4).array() = targets.row(i);
    bbox_weights.row(i).block(0, cls_ind * 4, 1, 4).array() = 1;
  }

  return std::make_tuple(b_rois, b_labels, bbox_targets, bbox_weights);
}
