#include "metrics.h"
#include "mxutils.h"

#include <mshadow/tensor.h>
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
  CheckMXnetError("RCNNAccMetric");
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
  std::vector<mx_float> pred_data(preds.Size());
  std::vector<mx_float> label_data(labels.Size());
  preds.SyncCopyToCPU(&pred_data, pred_data.size());
  labels.SyncCopyToCPU(&label_data, label_data.size());
  CheckMXnetError("RCNNLogLossMetric");

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
  auto value = cls_loss.sum();
  assert(!std::isnan(value));
  sum_metric += value;
  num_inst += label.rows();
}

void RCNNL1LossMetric::Update(mxnet::cpp::NDArray labels,
                              mxnet::cpp::NDArray preds) {
  std::vector<mx_float> pred_data(preds.Size());
  std::vector<mx_float> label_data(labels.Size());
  preds.SyncCopyToCPU(&pred_data, pred_data.size());
  labels.SyncCopyToCPU(&label_data, label_data.size());
  CheckMXnetError("RCNNL1LossMetric");

  auto bbox_loss =
      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>(
          pred_data.data(), static_cast<Eigen::Index>(pred_data.size()), 1);

  auto label = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic,
                                              Eigen::Dynamic, Eigen::RowMajor>>(
      label_data.data(), static_cast<Eigen::Index>(label_data.size()), 1);

  sum_metric += bbox_loss.array().sum();
  num_inst += (label.array() != 0).count();
}

void RPNL1LossMetric::Update(mxnet::cpp::NDArray labels,
                             mxnet::cpp::NDArray preds) {
  auto preds_shape = preds.GetShape();
  auto labels_shape = labels.GetShape();
  std::vector<mx_float> pred_data(preds.Size());
  std::vector<mx_float> label_data(labels.Size());
  preds.SyncCopyToCPU(&pred_data, pred_data.size());
  labels.SyncCopyToCPU(&label_data, label_data.size());
  CheckMXnetError("RPNL1LossMetric");

  auto bbox_loss =
      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>(
          pred_data.data(), static_cast<Eigen::Index>(pred_data.size()), 1);

  auto bbox_weight =
      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>(
          label_data.data(), static_cast<Eigen::Index>(label_data.size()), 1);

  // calculate num_inst(average on those fg anchors)
  auto num = (bbox_weight.array() > 0).count() / 4;
  auto val = bbox_loss.array().sum();
  sum_metric += val;
  num_inst += num;
}

void RPNAccMetric::Update(mxnet::cpp::NDArray labels,
                          mxnet::cpp::NDArray preds) {
  auto preds_shape = preds.GetShape();
  auto labels_shape = labels.GetShape();
  std::vector<mx_float> pred_data(labels.Size());
  std::vector<mx_float> label_data(labels.Size());
  preds.ArgmaxChannel().SyncCopyToCPU(&pred_data, pred_data.size());
  labels.SyncCopyToCPU(&label_data, label_data.size());
  CheckMXnetError("RPNAccMetric");
  for (mx_uint i = 0; i < labels.Size(); ++i) {
    if (static_cast<long>(label_data[i]) != -1) {
      sum_metric +=
          (static_cast<int>(pred_data[i]) == static_cast<int>(label_data[i]))
              ? 1
              : 0;
      num_inst += 1;
    }
  }
}

void RPNLogLossMetric::Update(mxnet::cpp::NDArray labels,
                              mxnet::cpp::NDArray preds) {
  using namespace mxnet::cpp;
  using namespace mshadow;
  std::vector<mx_float> label_data(labels.Size());
  labels.SyncCopyToCPU(&label_data, label_data.size());

  std::vector<mx_float> pred_data(preds.Size());
  preds.SyncCopyToCPU(&pred_data, pred_data.size());
  CheckMXnetError("RPNLogLossMetric");

  index_t labels_len = static_cast<index_t>(label_data.size());
  index_t len = static_cast<index_t>(pred_data.size());
  index_t dim3 = len / (preds.GetShape()[0] * preds.GetShape()[1]);
  auto out_shape = Shape2(labels_len, len / labels_len);
  Tensor<cpu, 2, float> pred(pred_data.data(), out_shape);
  {
    TensorContainer<cpu, 2, float> x3(out_shape);
    {
      Tensor<cpu, 4, float> pred_map(
          pred_data.data(), Shape4(preds.GetShape()[0], preds.GetShape()[1],
                                   preds.GetShape()[2], preds.GetShape()[3]));
      auto x1 = expr::reshape(
          pred_map, Shape3(preds.GetShape()[0], preds.GetShape()[1], dim3));
      auto x2 = expr::transpose(x1, Shape3(0, 2, 1));
      x3 = expr::reshape(x2, out_shape);
    }
    Copy(pred, x3);
  }

  // filter with labels
  std::vector<float> cls_data;
  cls_data.reserve(labels_len);
  for (index_t i = 0; i < labels_len; ++i) {
    if (static_cast<long>(label_data[i]) != -1) {
      cls_data.push_back(pred[i][static_cast<index_t>(label_data[i])]);
    }
  }

  Eigen::MatrixXf cls =
      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>(
          cls_data.data(), static_cast<Eigen::Index>(cls_data.size()), 1);

  cls.array() += 1e-14f;
  auto cls_loss = -1 * cls.array().log();
  sum_metric += cls_loss.sum();
  num_inst += cls_data.size();
}
