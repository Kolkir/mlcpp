#ifndef COMMON_H
#define COMMON_H

#include <mshadow/tensor.h>
#include <iostream>

template <typename Device>
struct ScopedTensorEngine {
  ScopedTensorEngine() { mshadow::InitTensorEngine<Device>(); }
  ~ScopedTensorEngine() { mshadow::ShutdownTensorEngine<Device>(); }
  ScopedTensorEngine(const ScopedTensorEngine&) = delete;
  ScopedTensorEngine& operator=(const ScopedTensorEngine&) = delete;
};

template <typename DType, typename Tensor>
void print_tensor(Tensor const& tensor, char const* label) {
  std::vector<DType> values(tensor.shape_.Size());
  mshadow::Tensor<mshadow::cpu, Tensor::kSubdim + 1, DType> cpu_tensor(
      values.data(), tensor.shape_);
  mshadow::Copy(cpu_tensor, tensor, tensor.stream_);
  std::cout << label << " :\n" << values << std::endl;
}

struct Pow {
  template <typename DType>
  MSHADOW_XINLINE static float Map(DType x, DType y) {
    return pow(x, y);
  }
};

struct Sqrt {
  template <typename DType>
  MSHADOW_XINLINE static float Map(DType x) {
    return sqrt(x);
  }
};

template <typename Device, typename DType>
auto get_min_max(mshadow::Tensor<Device, 2, DType> const& tensor) {
  using Tensor = mshadow::Tensor<Device, 2, DType>;
  mshadow::TensorContainer<Device, 1, DType> min(mshadow::Shape1(1));
  min.set_stream(tensor.stream_);
  mshadow::TensorContainer<Device, 1, DType> max(mshadow::Shape1(1));
  max.set_stream(tensor.stream_);

  min = mshadow::expr::ReduceTo1DExp<Tensor, DType, mshadow::red::minimum,
                                     mshadow::expr::ExpInfo<Tensor>::kDim - 1>(
      tensor, DType(1));
  max = mshadow::expr::ReduceTo1DExp<Tensor, DType, mshadow::red::maximum,
                                     mshadow::expr::ExpInfo<Tensor>::kDim - 1>(
      tensor, DType(1));

  mshadow::TensorContainer<mshadow::cpu, 1, DType> value(mshadow::Shape1(1));
  mshadow::Copy(value, min, tensor.stream_);
  DType v_min = value[0];
  mshadow::Copy(value, max, tensor.stream_);
  DType v_max = value[0];
  return std::make_pair(v_min, v_max);
}

template <typename Device, typename DType>
void load_data(std::vector<DType>& raw_data,
               mshadow::TensorContainer<Device, 2, DType>& dst) {
  mshadow::Tensor<mshadow::cpu, 2, DType> host_data(
      raw_data.data(), mshadow::Shape2(raw_data.size(), 1));
  dst.Resize(host_data.shape_);
  mshadow::Copy(dst, host_data, dst.stream_);
}

#endif  // COMMON_H
