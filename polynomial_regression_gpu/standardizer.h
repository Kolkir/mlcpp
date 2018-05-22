#ifndef STANDARDIZER_H
#define STANDARDIZER_H

#include <mshadow/tensor.h>
#include "common.h"

// Standardize 2D tensor of shape [rows]x[1]
template <typename Device, typename DType>
class Standardizer {
 public:
  using Tensor = mshadow::TensorContainer<Device, 2, DType>;
  using Stream = mshadow::Stream<Device>;

  Standardizer() {}
  ~Standardizer() {}
  Standardizer(const Standardizer&) = delete;
  Standardizer& operator=(const Standardizer&) = delete;

  void transform(Tensor& vec) {
    assert(vec.shape_.kDimension == 2);
    assert(vec.shape_[1] == 1);

    auto rows = vec.shape_[0];

    // alloc dst/temp tensors
    mean.Resize(mshadow::Shape1(1));
    mean.set_stream(vec.stream_);
    temp.Resize(mshadow::Shape2(rows, 1));
    temp.set_stream(vec.stream_);
    sd.Resize(mshadow::Shape1(1));
    sd.set_stream(vec.stream_);

    // calculate
    mean = mshadow::expr::sumall_except_dim<1>(vec);
    mean /= static_cast<DType>(rows);
    temp = mshadow::expr::F<Pow>(
        vec - mshadow::expr::broadcast<1>(mean, temp.shape_), 2);

    sd = mshadow::expr::sumall_except_dim<1>(temp);
    sd = mshadow::expr::F<Sqrt>(sd / static_cast<DType>(rows - 1));

    temp = (vec - mshadow::expr::broadcast<1>(mean, temp.shape_)) /
           mshadow::expr::broadcast<1>(sd, temp.shape_);

    mshadow::Copy(vec, temp, vec.stream_);
  }

  auto get_moments() {
    mshadow::TensorContainer<mshadow::cpu, 1, DType> value(mshadow::Shape1(1));
    mshadow::Copy(value, mean, mean.stream_);
    DType v_mean = value[0];
    mshadow::Copy(value, sd, sd.stream_);
    DType v_sd = value[0];
    return std::vector<DType>{v_mean, v_sd};
  }

 private:
  mshadow::TensorContainer<Device, 1, DType> mean;
  mshadow::TensorContainer<Device, 1, DType> sd;
  mshadow::TensorContainer<Device, 2, DType> temp;
};

#endif  // STANDARDIZER_H
