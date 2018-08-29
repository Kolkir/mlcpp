#include "resnet.h"

auto LoadPretrainedResnet(const mxnet::cpp::Context& ctx,
                          const std::string& param_file) {
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

static const double eps = 2e-5;
static const bool use_global_stats = true;
static const uint32_t workspace = 1024;

mxnet::cpp::Symbol ResidualUnit(mxnet::cpp::Symbol data,
                                uint32_t num_filter,
                                uint32_t stride,
                                bool dim_match,
                                const std::string& name) {
  using namespace mxnet::cpp;

  auto bn1 = Operator("BatchNorm")
                 .SetParam("fix_gamma", false)
                 .SetParam("eps", eps)
                 .SetParam("use_global_stats", use_global_stats)
                 .SetInput("data", data)
                 .CreateSymbol(name + "_bn1");

  auto act1 = Activation(name + "_relu1", bn1, "relu");

  auto conv1 = Operator("Convolution")
                   .SetParam("num_filter", static_cast<int>(num_filter * 0.25))
                   .SetParam("kernel", Shape(1, 1))
                   .SetParam("stride", Shape(1, 1))
                   .SetParam("pad", Shape(0, 0))
                   .SetParam("no_bias", true)
                   .SetParam("workspace", workspace)
                   .SetInput("data", act1)
                   .CreateSymbol(name + "_conv1");

  auto bn2 = Operator("BatchNorm")
                 .SetParam("fix_gamma", false)
                 .SetParam("eps", eps)
                 .SetParam("use_global_stats", use_global_stats)
                 .SetInput("data", conv1)
                 .CreateSymbol(name + "_bn2");

  auto act2 = Activation(name + "_relu2", bn1, "relu");

  auto conv2 = Operator("Convolution")
                   .SetParam("num_filter", static_cast<int>(num_filter * 0.25))
                   .SetParam("kernel", Shape(3, 3))
                   .SetParam("stride", Shape(stride, stride))
                   .SetParam("pad", Shape(1, 1))
                   .SetParam("no_bias", true)
                   .SetParam("workspace", workspace)
                   .SetInput("data", act2)
                   .CreateSymbol(name + "_conv2");

  auto bn3 = Operator("BatchNorm")
                 .SetParam("fix_gamma", false)
                 .SetParam("eps", eps)
                 .SetParam("use_global_stats", use_global_stats)
                 .SetInput("data", conv2)
                 .CreateSymbol(name + "_bn3");

  auto act3 = Activation(name + "_relu3", bn3, "relu");

  auto conv3 = Operator("Convolution")
                   .SetParam("num_filter", num_filter)
                   .SetParam("kernel", Shape(1, 1))
                   .SetParam("stride", Shape(1, 1))
                   .SetParam("pad", Shape(0, 0))
                   .SetParam("no_bias", true)
                   .SetParam("workspace", workspace)
                   .SetInput("data", act3)
                   .CreateSymbol(name + "_conv3");

  Symbol shortcut;

  if (dim_match) {
    shortcut = data;
  } else {
    Operator("Convolution")
        .SetParam("num_filter", num_filter)
        .SetParam("kernel", Shape(1, 1))
        .SetParam("stride", Shape(stride, stride))
        .SetParam("no_bias", true)
        .SetParam("workspace", workspace)
        .SetInput("data", act1)
        .CreateSymbol(name + "_sc");
  }

  auto sum = Operator("ElementWiseSum")
                 .SetParam("num_args", 2)
                 .SetInput("a", conv3)
                 .SetInput("b", shortcut)
                 .CreateSymbol(name + "_plus");
  return sum;
}

mxnet::cpp::Symbol GetResnetHeadSymbol(
    mxnet::cpp::Symbol data,
    const std::vector<uint32_t>& units,
    const std::vector<uint32_t>& filter_list) {
  using namespace mxnet::cpp;
  // res1
  auto data_bn = Operator("BatchNorm")
                     .SetParam("fix_gamma", true)
                     .SetParam("eps", eps)
                     .SetParam("use_global_stats", use_global_stats)
                     .SetInput("data", data)
                     .CreateSymbol("bn_data");

  auto conv0 = Operator("Convolution")
                   .SetParam("num_filter", 64)
                   .SetParam("kernel", Shape(7, 7))
                   .SetParam("stride", Shape(2, 2))
                   .SetParam("pad", Shape(3, 3))
                   .SetParam("no_bias", true)
                   .SetParam("workspace", workspace)
                   .SetInput("data", data_bn)
                   .CreateSymbol("conv0");

  auto bn0 = Operator("BatchNorm")
                 .SetParam("fix_gamma", false)
                 .SetParam("eps", eps)
                 .SetParam("use_global_stats", use_global_stats)
                 .SetInput("data", conv0)
                 .CreateSymbol("bn0");

  auto relu0 = Activation("relu0", bn0, "relu");

  auto pool0 =
      Pooling("pool0", relu0, Shape(3, 3), PoolingPoolType::kMax, false, false,
              PoolingPoolingConvention::kValid, Shape(2, 2), Shape(1, 1));

  // res2
  auto unit = ResidualUnit(pool0, filter_list[0], 1, false, "stage1_unit1");
  for (uint32_t i = 2; i < units[0] + 1; ++i) {
    std::stringstream name;
    name << "stage1_unit" << i << "s";
    unit = ResidualUnit(unit, filter_list[0], 1, true, name.str());
  }

  // res3
  unit = ResidualUnit(unit, filter_list[1], 2, false, "stage2_unit1");
  for (uint32_t i = 2; i < units[1] + 1; ++i) {
    std::stringstream name;
    name << "stage2_unit" << i << "s";
    unit = ResidualUnit(unit, filter_list[1], 1, true, name.str());
  }

  // res4
  unit = ResidualUnit(unit, filter_list[2], 2, false, "stage3_unit1");
  for (uint32_t i = 2; i < units[2] + 1; ++i) {
    std::stringstream name;
    name << "stage3_unit" << i << "s";
    unit = ResidualUnit(unit, filter_list[2], 1, true, name.str());
  }

  return unit;
}

mxnet::cpp::Symbol GetResnetTopSymbol(
    mxnet::cpp::Symbol data,
    const std::vector<uint32_t>& units,
    const std::vector<uint32_t>& filter_list) {
  using namespace mxnet::cpp;
  auto unit = ResidualUnit(data, filter_list[3], 2, false, "stage4_unit1");
  for (uint32_t i = 2; i < units[3] + 1; ++i) {
    std::stringstream name;
    name << "stage4_unit" << i << "s";
    unit = ResidualUnit(unit, filter_list[3], 1, true, name.str());
  }
  auto bn1 = Operator("BatchNorm")
                 .SetParam("fix_gamma", false)
                 .SetParam("eps", eps)
                 .SetParam("use_global_stats", use_global_stats)
                 .SetInput("data", unit)
                 .CreateSymbol("bn1");
  auto relu1 = Activation("relu1", bn1, "relu");

  auto pool1 =
      Pooling("pool1", relu1, Shape(7, 7), PoolingPoolType::kAvg, true);

  return pool1;
}
