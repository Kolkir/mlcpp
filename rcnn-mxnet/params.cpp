#include "params.h"

std::pair<std::map<std::string, mxnet::cpp::NDArray>,
          std::map<std::string, mxnet::cpp::NDArray>>
LoadNetParams(const mxnet::cpp::Context& ctx, const std::string& param_file) {
  using namespace mxnet::cpp;
  //---------- Load parameters
  std::map<std::string, NDArray> paramters;
  NDArray::Load(param_file, nullptr, &paramters);
  std::map<std::string, NDArray> args_map;
  std::map<std::string, NDArray> aux_map;
  for (const auto& k : paramters) {
    if (k.first.substr(0, 4) == "aux:") {
      auto name = k.first.substr(4, k.first.size() - 4);
      aux_map[name] = k.second.Copy(ctx);
    }
    if (k.first.substr(0, 4) == "arg:") {
      auto name = k.first.substr(4, k.first.size() - 4);
      args_map[name] = k.second.Copy(ctx);
    }
  }
  /*WaitAll is need when we copy data between GPU and the main memory*/
  NDArray::WaitAll();
  return std::make_pair(args_map, aux_map);
}

void SaveNetParams(const std::string& param_file, mxnet::cpp::Executor* exe) {
  std::map<std::string, mxnet::cpp::NDArray> params;
  for (auto& iter : exe->arg_dict()) {
    if (iter.first.rfind("data", 0) != 0 &&
        iter.first.rfind("im_info", 0) != 0 &&
        iter.first.rfind("gt_boxes", 0) != 0 &&
        iter.first.rfind("label", 0) != 0 &&
        iter.first.rfind("bbox_target", 0) != 0 &&
        iter.first.rfind("bbox_weight", 0) != 0)
      params.insert({"arg:" + iter.first, iter.second});
  }
  for (auto iter : exe->aux_dict()) {
    params.insert({"aux:" + iter.first, iter.second});
  }
  mxnet::cpp::NDArray::Save(param_file, params);
}
