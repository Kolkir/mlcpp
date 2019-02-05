#include "nnutils.h"
#include "debug.h"

#include <cmath>

SamePad2dImpl::SamePad2dImpl() {}

SamePad2dImpl::SamePad2dImpl(uint32_t kernel_size, uint32_t stride)
    : kernel_size_(kernel_size), stride_(stride) {}

/* Makes output size is the same as input size,
 * after applying specified kernel and stride
 *              pad|                                      |pad
 *  inputs       0 |1  2  3  4  5  6  7  8  9  10 11 12 13|0  0
 *              |________________|
 *                             |_________________|
 *                                            |________________|
 */
at::Tensor SamePad2dImpl::forward(at::Tensor input) {
  assert(input.ndimension() == 4);
  assert(stride_ != 0);
  auto in_width = input.size(2);
  auto in_height = input.size(3);
  auto out_width =
      std::ceil(static_cast<float>(in_width) / static_cast<float>(stride_));
  auto out_height =
      std::ceil(static_cast<float>(in_height) / static_cast<float>(stride_));
  auto pad_along_width = ((out_width - 1) * stride_ + kernel_size_ - in_width);
  auto pad_along_height =
      ((out_height - 1) * stride_ + kernel_size_ - in_height);

  auto pad_left = static_cast<uint32_t>(std::floor(pad_along_width / 2));
  auto pad_top = static_cast<uint32_t>(std::floor(pad_along_height / 2));
  auto pad_right = static_cast<uint32_t>(pad_along_width - pad_left);
  auto pad_bottom = static_cast<uint32_t>(pad_along_height - pad_top);

  at::Tensor top = at::zeros(
      {input.size(0), input.size(1), input.size(2), pad_top}, input.type());
  at::Tensor bottom = at::zeros(
      {input.size(0), input.size(1), input.size(2), pad_bottom}, input.type());
  input = at::cat({top, input, bottom}, 3);

  at::Tensor left = at::zeros(
      {input.size(0), input.size(1), pad_left, input.size(3)}, input.type());
  at::Tensor right = at::zeros(
      {input.size(0), input.size(1), pad_right, input.size(3)}, input.type());
  input = at::cat({left, input, right}, 2);

  return input;
}

at::Tensor upsample(at::Tensor x, float scale_factor) {
  auto output_size = [scale_factor, &x](uint32_t dim) {
    std::vector<int64_t> sizes(dim);
    for (size_t i = 0; i < dim; ++i) {
      sizes[i] = static_cast<int64_t>(
          std::floor(x.size(static_cast<int64_t>(i) + 2) * scale_factor));
    }
    return sizes;
  };

  std::vector<int64_t> sizes;
  if (x.ndimension() == 3) {
    sizes = output_size(1);
    return torch::upsample_nearest1d(x, torch::IntList(sizes));
  } else if (x.ndimension() == 4) {
    sizes = output_size(2);
    return torch::upsample_nearest2d(x, torch::IntList(sizes));
  } else if (x.ndimension() == 5) {
    sizes = output_size(3);
    return torch::upsample_nearest3d(x, torch::IntList(sizes));
  } else {
    throw std::invalid_argument(
        "Upsamle Input Error: Only 3D, 4D and 5D input Tensors supported : " +
        std::to_string(x.ndimension()));
  }
}

at::Tensor unique1d(at::Tensor tensor) {
  if (tensor.size(0) == 0 || tensor.size(0) == 1)
    return tensor;
  std::tie(tensor, std::ignore) = tensor.sort();
  auto unique_bool = tensor.narrow(0, 1, tensor.size(0) - 1) !=
                     tensor.narrow(0, 0, tensor.size(0) - 1);
  auto first_element =
      torch::tensor({1}, at::dtype(at::kByte).requires_grad(false));
  if (tensor.is_cuda())
    first_element = first_element.cuda();
  unique_bool = torch::cat({first_element, unique_bool}, /*dim*/ 0);
  return tensor.masked_select(unique_bool);
}

at::Tensor intersect1d(at::Tensor tensor1, at::Tensor tensor2) {
  auto aux = torch::cat({tensor1, tensor2}, /*dim*/ 0);
  if (!is_empty(aux)) {
    std::tie(aux, std::ignore) = aux.sort();
    auto unique_bool =
        aux.narrow(0, 1, aux.size(0) - 1) == aux.narrow(0, 0, aux.size(0) - 1);
    return aux.narrow(0, 0, aux.size(0) - 1).masked_select(unique_bool);
  } else {
    return aux;
  }
}

void ClipGradNorm(std::vector<at::Tensor> parameters, float max_norm) {
  double total_norm = 0.0;
  for (auto& p : parameters) {
    if (p.requires_grad()) {
      auto param_norm = p.grad();
      if (!is_empty(param_norm)) {
        param_norm = param_norm.norm();
        total_norm += std::pow(param_norm.item<float>(), 2.f);
      }
    }
  }
  total_norm = std::pow(total_norm, (1. / 2.));
  auto clip_coef = max_norm / (total_norm + 1e-6);
  if (clip_coef < 1) {
    for (auto& p : parameters) {
      auto param_norm = p.grad();
      if (!is_empty(param_norm)) {
        p.grad().mul_(clip_coef);
      }
    }
  }
}

bool is_empty(at::Tensor x) {
  if (x.defined() && x.dim() > 0 && x.size(0) != 0 && x.numel() > 0)
    return false;
  else
    return true;
}
