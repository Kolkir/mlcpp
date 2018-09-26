#include "metrics.h"

#include <Eigen/Dense>

void RCNNAccMetric::Update(mxnet::cpp::NDArray labels,
                           mxnet::cpp::NDArray preds) {
  auto preds_shape = preds.GetShape();
  auto labels_shape = labels.GetShape();
  mx_uint batches = labels_shape.front();
  mx_uint len = labels_shape.back() * batches;
  mx_uint classes_num = preds_shape.back();
  std::vector<mx_float> pred_data(len);
  std::vector<mx_float> label_data(len);
  preds.Reshape(mxnet::cpp::Shape(static_cast<mx_uint>(-1), classes_num))
      .ArgmaxChannel()
      .SyncCopyToCPU(&pred_data, pred_data.size());
  labels.Reshape(mxnet::cpp::Shape(static_cast<mx_uint>(-1)))
      .SyncCopyToCPU(&label_data, label_data.size());
  for (mx_uint i = 0; i < len; ++i) {
    sum_metric +=
        (static_cast<int>(pred_data[i]) == static_cast<int>(label_data[i])) ? 1
                                                                            : 0;
    num_inst += 1;
  }
}

void RCNNLogLossMetric::Update(mxnet::cpp::NDArray labels,
                               mxnet::cpp::NDArray preds) {
  auto preds_shape = preds.GetShape();
  auto labels_shape = labels.GetShape();
  mx_uint batches = labels_shape.front();
  mx_uint len = labels_shape.back() * batches;
  mx_uint classes_num = preds_shape.back();
  std::vector<mx_float> pred_data(len * classes_num);
  std::vector<mx_float> label_data(len);
  preds.Reshape(mxnet::cpp::Shape(static_cast<mx_uint>(-1), classes_num))
      .SyncCopyToCPU(&pred_data, pred_data.size());
  labels.Reshape(mxnet::cpp::Shape(static_cast<mx_uint>(-1)))
      .SyncCopyToCPU(&label_data, label_data.size());

  auto pred = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>>(
      pred_data.data(), len, classes_num);

  auto label = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic,
                                              Eigen::Dynamic, Eigen::RowMajor>>(
      label_data.data(), len, 1);

  Eigen::MatrixXf cls(len, 1);
  for (Eigen::Index r = 0; r < len; ++r) {
    cls(r, 0) = pred(r, static_cast<Eigen::Index>(label(r, 0)));
  }

  cls.array() += 1e-14f;
  auto cls_loss = -1 * cls.array().log();
  sum_metric += cls_loss.sum();
  num_inst += label.rows();
}
