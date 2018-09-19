#include "metrics.h"

void RCNNAccMetric::Update(mxnet::cpp::NDArray labels,
                           mxnet::cpp::NDArray preds) {
  auto preds_shape = preds.GetShape();
  auto labels_shape = labels.GetShape();
  mx_uint len = labels_shape.back();
  mx_uint classes_num = preds_shape.back();
  std::vector<mx_float> pred_data(len);
  std::vector<mx_float> label_data(len);
  preds.Reshape(mxnet::cpp::Shape(static_cast<mx_uint>(-1), classes_num))
      .ArgmaxChannel()
      .SyncCopyToCPU(&pred_data, len);
  labels.Reshape(mxnet::cpp::Shape(static_cast<mx_uint>(-1)))
      .SyncCopyToCPU(&label_data, len);
  for (mx_uint i = 0; i < len; ++i) {
    sum_metric += (pred_data[i] == label_data[i]) ? 1 : 0;
    num_inst += 1;
  }
}
