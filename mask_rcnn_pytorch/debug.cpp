#ifndef NDEBUG

#include "debug.h"
#include <algorithm>
#include <iostream>

TensorInfo __attribute__((used, noinline)) PrintTensor(at::Tensor val) {
  return TensorInfo(val);
}

TensorInfo::TensorInfo(at::Tensor tensor) {
  auto sizes = tensor.sizes();
  auto dim_number = sizes.size();
  if (dim_number > 0) {
    dims.resize(dim_number);
    std::copy(sizes.begin(), sizes.end(), dims.begin());
    is_variable = tensor.is_variable();
    const auto type_name = tensor.dtype().name();
    type = type_name;
  }
}

#endif
