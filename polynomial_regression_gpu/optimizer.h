#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <mshadow/tensor.h>
#include "common.h"

#include <iostream>

template <typename Device, typename DType>
class Optimizer {
 public:
  Optimizer() {}

  void predict(mshadow::Tensor<Device, 2, DType> const& x,
               mshadow::Tensor<Device, 2, DType>& y) {
    y = mshadow::expr::dot(x, gpu_weights);
  }

  void fit(mshadow::Tensor<Device, 2, DType> const& x,
           mshadow::Tensor<Device, 2, DType> const& y) {
    assert(y.shape_.kDimension == 2);
    assert(y.shape_[1] == 1);
    assert(x.shape_.kDimension == 2);
    assert(x.shape_[0] == y.shape_[0]);

    size_t cols = x.shape_[1];
    size_t rows = x.shape_[0];
    size_t n_batches = rows / batch_size;

    // it is important to allocate all tensors before assiging
    gpu_weights.set_stream(x.stream_);
    gpu_weights.Resize(mshadow::Shape2(cols, 1));
    gpu_weights = 0.0f;

    gpu_grad.Resize(mshadow::Shape2(cols, 1));
    gpu_grad.set_stream(x.stream_);

    yhat.Resize(mshadow::Shape2(batch_size, 1));
    yhat.set_stream(x.stream_);

    error.Resize(mshadow::Shape2(batch_size, 1));
    error.set_stream(x.stream_);

    error_total.Resize(mshadow::Shape2(rows, 1));
    error_total.set_stream(x.stream_);

    error_total_cpu.Resize(mshadow::Shape2(rows, 1));

    gpu_eg_sum.Resize(mshadow::Shape2(cols, 1));
    gpu_eg_sum.set_stream(x.stream_);
    gpu_eg_sum = 0.f;

    gpu_weights_delta.Resize(mshadow::Shape2(cols, 1));
    gpu_weights_delta.set_stream(x.stream_);

    gpu_ex_sum.Resize(mshadow::Shape2(cols, 1));
    gpu_ex_sum.set_stream(x.stream_);
    gpu_ex_sum = 0.f;

    // gradient descent
    for (size_t epoch = 0; epoch < n_epochs; ++epoch) {
      for (size_t bi = 0; bi < n_batches; ++bi) {
        auto bs = bi * batch_size;
        auto be = bs + batch_size;
        auto batch_x = x.Slice(bs, be);
        auto batch_y = y.Slice(bs, be);

        // Print weights
        // print_tensor<DType>(gpu_weights, "weights");

        predict(batch_x, yhat);

        error = yhat - batch_y;
        gpu_grad = mshadow::expr::dot(batch_x.T(), error);
        gpu_grad /= batch_size;

        // print_tensor<DType>(gpu_grad, "grad");
        // print_tensor<DType>(error, "error");

        // AdaDelta
        gpu_eg_sum =
            lr * gpu_eg_sum + (1.f - lr) * mshadow::expr::F<Pow>(gpu_grad, 2);
        gpu_weights_delta = -1.f *
                            (mshadow::expr::F<Sqrt>(gpu_ex_sum + e) /
                             mshadow::expr::F<Sqrt>(gpu_eg_sum + e)) *
                            gpu_grad;
        gpu_ex_sum = lr * gpu_ex_sum +
                     (1.f - lr) * mshadow::expr::F<Pow>(gpu_weights_delta, 2);
        gpu_weights = gpu_weights + gpu_weights_delta;

        // BGD
        // gpu_weights = gpu_weights - (lr * gpu_grad);
      }
      // compute cost
      error_total = mshadow::expr::dot(x, gpu_weights);
      error_total = mshadow::expr::F<Pow>(error_total - y, 2);

      mshadow::Copy(error_total_cpu, error_total, error_total.stream_);
      long double cost = 0;
      for (size_t r = 0; r < rows; ++r)
        cost += error_total_cpu[r][0];
      cost /= rows;
      std::cout << "Epoch " << epoch << " cost = " << cost << std::endl;
    }
  }

 private:
  size_t n_epochs = 5000;
  size_t batch_size = 8;
  // DType lr = 0.01;  // BGD

  mshadow::TensorContainer<Device, 2, DType> gpu_weights;
  mshadow::TensorContainer<Device, 2, DType> gpu_grad;
  mshadow::TensorContainer<Device, 2, DType> yhat;
  mshadow::TensorContainer<Device, 2, DType> error;
  mshadow::TensorContainer<Device, 2, DType> error_total;
  mshadow::TensorContainer<mshadow::cpu, 2, DType> error_total_cpu;

  // AdaDelta
  DType lr = 0.99;
  DType e = std::numeric_limits<DType>::epsilon();
  mshadow::TensorContainer<Device, 2, DType> gpu_eg_sum;
  mshadow::TensorContainer<Device, 2, DType> gpu_weights_delta;
  mshadow::TensorContainer<Device, 2, DType> gpu_ex_sum;
};

#endif  // OPTIMIZER_H
