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
