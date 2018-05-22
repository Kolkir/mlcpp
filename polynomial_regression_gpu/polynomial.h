#ifndef POLYNOM_H
#define POLYNOM_H

#include <mshadow/tensor.h>
#include <algorithm>
#include "common.h"
#include "standardizer.h"

template <typename DType>
std::pair<DType, DType> get_range_for_degree(DType degree) {
  auto max_v = static_cast<DType>(
      std::pow(std::numeric_limits<DType>::max(), 1. / (degree - 1)));
  return std::make_pair(-max_v, max_v);
}

// takes tensor N x 1 and returns tensor N x p_degree
template <typename Device, typename DType>
void generate_polynomial(mshadow::Tensor<Device, 2, DType> const& tensor,
                         mshadow::TensorContainer<Device, 2, DType> const& poly,
                         size_t p_degree) {
  assert(tensor.shape_.kDimension == 2);
  assert(tensor.shape_[1] == 1);

  auto rows = tensor.shape_[0];
  mshadow::TensorContainer<Device, 2, DType> col_temp(mshadow::Shape2(rows, 1));
  col_temp.set_stream(tensor.stream_);

  for (size_t c = 0; c < p_degree; ++c) {
    auto col = mshadow::expr::slice(poly, mshadow::Shape2(0, c),
                                    mshadow::Shape2(rows, c + 1));
    col_temp = mshadow::expr::F<Pow>(tensor, static_cast<DType>(c));
    // print_tensor<DType>(col_temp, "pow");
    col = col_temp;
  }
}

#endif  // POLYNOM_H
