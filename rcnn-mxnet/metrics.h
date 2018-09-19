#ifndef METRICS_H
#define METRICS_H

#include <mxnet-cpp/MxNetCpp.h>

class RCNNAccMetric : public mxnet::cpp::EvalMetric {
 public:
  RCNNAccMetric() : EvalMetric("RCNNAccMetric") {}

  void Update(mxnet::cpp::NDArray labels, mxnet::cpp::NDArray preds) override;
};

#endif  // METRICS_H
