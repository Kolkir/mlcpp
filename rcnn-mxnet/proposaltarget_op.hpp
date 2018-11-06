#ifndef PROPOSALTARGET_OP_HPP
#define PROPOSALTARGET_OP_HPP

#include "bbox.h"
#include "proposaltarget_op.h"

#include <Eigen/Dense>
#include <tuple>

namespace mxnet {
namespace op {

template <typename xpu>
class ProposalTargetOp : public Operator {
 public:
  explicit ProposalTargetOp(ProposalTargetParam param) { this->param_ = param; }

  virtual void Forward(const OpContext& ctx,
                       const std::vector<TBlob>& in_data,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& out_data,
                       const std::vector<TBlob>& /*aux_states*/) override {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(param_.batch_images, in_data[1].shape_[0]);
    using namespace mshadow;
    mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();

    // get input tensors
    // rois [n, 5] (batch_index, x1, y1, x2, y2)
    Tensor<xpu, 2> all_rois_x = in_data[0].get<xpu, 2, real_t>(s);
    // gt_boxes [b, n, 5] (x1, y1, x2, y2, cls)
    Tensor<xpu, 3> all_gt_boxes_x = in_data[1].get<xpu, 3, real_t>(s);

    // define result shapes
    auto rois_shape = Shape2(static_cast<index_t>(param_.batch_rois), 5);
    auto label_shape = Shape1(static_cast<index_t>(param_.batch_rois));
    auto bbox_target_shape =
        Shape2(static_cast<index_t>(param_.batch_rois),
               static_cast<index_t>(param_.num_classes * 4));
    auto bbox_weight_shape =
        Shape2(static_cast<index_t>(param_.batch_rois),
               static_cast<index_t>(param_.num_classes * 4));

    // ---------------- Main logic - cpu version ----------------------------
    auto rois_per_image = param_.batch_rois / param_.batch_images;
    auto fg_rois_per_image =
        static_cast<int>(std::round(param_.fg_fraction * rois_per_image));
    std::vector<float> box_stds;
    box_stds.assign(param_.box_stds.begin(), param_.box_stds.end());

    using EigenMatrix =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    EigenMatrix rois(rois_shape[0], rois_shape[1]);
    EigenMatrix labels(label_shape[0], 1);
    EigenMatrix bbox_targets(bbox_target_shape[0], bbox_target_shape[1]);
    EigenMatrix bbox_weights(bbox_weight_shape[0], bbox_weight_shape[1]);

    EigenMatrix all_rois(all_rois_x.shape_[0], all_rois_x.shape_[1]);
    Tensor<cpu, 2, real_t> all_rois_t(all_rois.data(), all_rois_x.shape_);
    Copy(all_rois_t, all_rois_x, s);

    EigenMatrix batch_gt_boxes(all_gt_boxes_x.shape_[1],
                               all_gt_boxes_x.shape_[2]);
    for (int batch_idx = 0; batch_idx < param_.batch_images; ++batch_idx) {
      //----------------------------------------------
      // select gt boxes related to the current batch element
      Tensor<cpu, 2, real_t> all_gt_boxes_t(
          batch_gt_boxes.data(),
          Shape2(all_gt_boxes_x.shape_[1], all_gt_boxes_x.shape_[2]));
      Copy(all_gt_boxes_t, all_gt_boxes_x[static_cast<index_t>(batch_idx)], s);

      // select gt boxes with foreground class
      auto gt_boxes_count = (batch_gt_boxes.col(4).array() > 0).count();
      Eigen::MatrixXf gt_boxes(gt_boxes_count, all_gt_boxes_x.shape_[2]);
      for (Eigen::Index i = 0, j = 0; i < batch_gt_boxes.rows();
           ++i) {  // col major
        if (batch_gt_boxes(i, 4) > 0)
          gt_boxes.row(j++) = batch_gt_boxes.row(i);
      }
      //----------------------------------------------
      // select rois related to the current batch element, images have different
      // size so rois count can also be different
      auto batch_rois_count =
          (all_rois.col(0).array() == static_cast<float>(batch_idx)).count();

      Eigen::MatrixXf batch_rois(
          batch_rois_count,
          all_rois.cols() - 1);  // exclude batch index col
      for (Eigen::Index i = 0, j = 0; i < all_rois.rows(); ++i) {  // col major
        if (static_cast<int>(all_rois(i, 0)) == batch_idx)
          batch_rois.row(j++) = all_rois.row(i).rightCols(4);
      }

      //----------------------------------------------
      // Include ground-truth boxes in the set of candidate rois
      {
        Eigen::MatrixXf rois_and_gt(batch_rois.rows() + gt_boxes.rows(),
                                    rois.cols() - 1);
        rois_and_gt << batch_rois,
            gt_boxes.block(0, 0, gt_boxes.rows(), gt_boxes.cols() - 1);
        batch_rois = rois_and_gt;
      }

      //----------------------------------------------
      // mark rois for this batch
      rois.block(batch_idx * rois_per_image, 0, rois_per_image, 1).array() =
          batch_idx;

      //----------------------------------------------
      // generate random sample of ROIs comprising foreground and background
      // examples
      auto b_rois = rois.block(batch_idx * rois_per_image, 1, rois_per_image,
                               rois.cols() - 1);
      auto b_labels = labels.block(batch_idx * rois_per_image, 0,
                                   rois_per_image, labels.cols());
      auto b_bbox_targets = bbox_targets.block(
          batch_idx * rois_per_image, 0, rois_per_image, bbox_targets.cols());
      auto b_bbox_weights = bbox_weights.block(
          batch_idx * rois_per_image, 0, rois_per_image, bbox_weights.cols());
      std::tie(b_rois, b_labels, b_bbox_targets, b_bbox_weights) =
          SampleRois(batch_rois, gt_boxes, param_.num_classes, rois_per_image,
                     fg_rois_per_image, param_.fg_overlap, box_stds);
    }

    Tensor<cpu, 2, real_t> rois_t(rois.data(), rois_shape);
    Tensor<cpu, 1, real_t> labels_t(labels.data(), label_shape);
    Tensor<cpu, 2, real_t> bbox_targets_t(bbox_targets.data(),
                                          bbox_target_shape);
    Tensor<cpu, 2, real_t> bbox_weights_t(bbox_weights.data(),
                                          bbox_weight_shape);

    // ---------------- Main logic end --------------------------

    // get destination tensors
    Tensor<xpu, 2> out0 =
        out_data[0].get_with_shape<xpu, 2, real_t>(rois_shape, s);
    Tensor<xpu, 1> out1 =
        out_data[1].get_with_shape<xpu, 1, real_t>(label_shape, s);
    Tensor<xpu, 2> out2 =
        out_data[2].get_with_shape<xpu, 2, real_t>(bbox_target_shape, s);
    Tensor<xpu, 2> out3 =
        out_data[3].get_with_shape<xpu, 2, real_t>(bbox_weight_shape, s);

    // copy results
    for (auto r : req) {
      CHECK_EQ(kWriteTo, r);
    }

    Copy(out0, rois_t, s);
    Copy(out1, labels_t, s);
    Copy(out2, bbox_targets_t, s);
    Copy(out3, bbox_weights_t, s);
  }

  void Backward(const OpContext& ctx,
                const std::vector<TBlob>& /*out_grad*/,
                const std::vector<TBlob>& /*in_data*/,
                const std::vector<TBlob>& /*out_data*/,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& in_grad,
                const std::vector<TBlob>& /*aux_states*/) override {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_grad.size(), 2);

    Stream<xpu>* s = ctx.get_stream<xpu>();
    auto grad0 = in_grad[0].get<xpu, 2, real_t>(s);
    auto grad1 = in_grad[1].get<xpu, 3, real_t>(s);

    // can not assume the grad would be zero
    Assign(grad0, req[0], 0);
    Assign(grad1, req[1], 0);
  }

 private:
  ProposalTargetParam param_;
};  // class ProposalOp

}  // namespace op
}  // namespace mxnet
#endif  // PROPOSALTARGET_OP_HPP
