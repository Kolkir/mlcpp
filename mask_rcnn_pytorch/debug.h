#ifndef DEBUG_H
#define DEBUG_H

#ifndef NDEBUG

#include <torch/torch.h>
#include <string>
#include <utility>
#include <vector>

class TensorInfo {
 public:
  TensorInfo(at::Tensor tensor);

  ~TensorInfo() {}

  TensorInfo(const TensorInfo& info) { Copy(info); }

  TensorInfo& operator=(const TensorInfo& info) {
    Copy(info);
    return *this;
  }

  TensorInfo(TensorInfo&& info) { Move(std::forward<TensorInfo>(info)); }

  TensorInfo& operator=(TensorInfo&& info) {
    Move(std::forward<TensorInfo>(info));
    return *this;
  }

 private:
  void Move(TensorInfo&& info) {
    if (this != &info) {
      dims = std::move(info.dims);
      type = std::move(info.type);
      is_variable = info.is_variable;
    }
  }

  void Copy(const TensorInfo& info) {
    if (this != &info) {
      dims = info.dims;
      type = info.type;
      is_variable = info.is_variable;
    }
  }

 public:
  bool is_variable{false};
  std::string type;
  std::vector<int64_t> dims;
};

TensorInfo __attribute__((used, noinline)) PrintTensor(at::Tensor val);

#endif

#endif  // DEBUG_H
