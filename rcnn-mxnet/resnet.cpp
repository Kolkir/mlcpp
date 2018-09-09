#include "resnet.h"

static const double eps = 2e-5;
static const bool use_global_stats = true;
static const uint64_t workspace = 1024;

mxnet::cpp::Symbol BatchNormUnit(mxnet::cpp::Symbol data,
                                 const std::string& name,
                                 bool fix_gamma) {
  using namespace mxnet::cpp;
  Symbol gamma(name + "_gamma");
  Symbol beta(name + "_beta");
  Symbol mmean(name + "_moving_mean");
  Symbol mvar(name + "_moving_var");

  return BatchNorm(name, data, gamma, beta, mmean, mvar, eps, 0.9f, fix_gamma,
                   use_global_stats);
}

mxnet::cpp::Symbol ConvolutionUnit(mxnet::cpp::Symbol data,
                                   const std::string& name,
                                   uint32_t num_filter,
                                   mxnet::cpp::Shape kernel,
                                   mxnet::cpp::Shape stride,
                                   mxnet::cpp::Shape pad) {
  using namespace mxnet::cpp;
  Symbol weight(name + "_weight");
  // Symbol bias(name + "_bias");

  // Why doesn't work ? - Convolution no-bias assume 2 inputs
  //  return Convolution(name, data, weight, bias, kernel, num_filter, stride,
  //                     Shape(1, 1), pad, 1, workspace, true);

  return Operator("Convolution")
      .SetParam("kernel", kernel)
      .SetParam("num_filter", num_filter)
      .SetParam("stride", stride)
      .SetParam("dilate", Shape(1, 1))
      .SetParam("pad", pad)
      .SetParam("num_group", 1)
      .SetParam("workspace", workspace)
      .SetParam("no_bias", true)
      .SetInput("data", data)
      .SetInput("weight", weight)
      .CreateSymbol(name);
}

mxnet::cpp::Symbol ResidualUnit(mxnet::cpp::Symbol data,
                                uint32_t num_filter,
                                uint32_t stride,
                                bool dim_match,
                                const std::string& name) {
  using namespace mxnet::cpp;

  auto bn1 = BatchNormUnit(data, name + "_bn1", false);

  auto act1 = Activation(name + "_relu1", bn1, "relu");

  auto conv1 = ConvolutionUnit(act1, name + "_conv1",
                               static_cast<uint32_t>(num_filter * 0.25),
                               Shape(1, 1), Shape(1, 1), Shape(0, 0));

  auto bn2 = BatchNormUnit(conv1, name + "_bn2", false);

  auto act2 = Activation(name + "_relu2", bn2, "relu");

  auto conv2 = ConvolutionUnit(act2, name + "_conv2",
                               static_cast<uint32_t>(num_filter * 0.25),
                               Shape(3, 3), Shape(stride, stride), Shape(1, 1));

  auto bn3 = BatchNormUnit(conv2, name + "_bn3", false);

  auto act3 = Activation(name + "_relu3", bn3, "relu");

  auto conv3 = ConvolutionUnit(act3, name + "_conv3", num_filter, Shape(1, 1),
                               Shape(1, 1), Shape(0, 0));

  Symbol shortcut;

  if (dim_match) {
    shortcut = data;
  } else {
    shortcut = ConvolutionUnit(act1, name + "_sc", num_filter, Shape(1, 1),
                               Shape(stride, stride), Shape(0, 0));
  }

  //  auto sum = Operator("ElementWiseSum")
  //                 .SetParam("num_args", 2)
  //                 .SetInput("a", conv3)
  //                 .SetInput("b", shortcut)
  //                 .CreateSymbol(name + "_plus");
  Symbol sum = shortcut + conv3;
  return sum;
}

mxnet::cpp::Symbol GetResnetHeadSymbol(
    mxnet::cpp::Symbol data,
    const std::vector<uint32_t>& units,
    const std::vector<uint32_t>& filter_list) {
  using namespace mxnet::cpp;
  // res1
  auto data_bn = BatchNormUnit(data, "bn_data", true);

  auto conv0 = ConvolutionUnit(data_bn, "conv0", 64, Shape(7, 7), Shape(2, 2),
                               Shape(3, 3));

  auto bn0 = BatchNormUnit(conv0, "bn0", false);

  auto relu0 = Activation("relu0", bn0, "relu");

  auto pool0 =
      Pooling("pool0", relu0, Shape(3, 3), PoolingPoolType::kMax, false, false,
              PoolingPoolingConvention::kValid, Shape(2, 2), Shape(1, 1));
  // res2
  auto unit = ResidualUnit(pool0, filter_list[0], 1, false, "stage1_unit1");
  for (uint32_t i = 2; i < units[0] + 1; ++i) {
    std::stringstream name;
    name << "stage1_unit" << i;
    unit = ResidualUnit(unit, filter_list[0], 1, true, name.str());
  }

  // res3
  unit = ResidualUnit(unit, filter_list[1], 2, false, "stage2_unit1");
  for (uint32_t i = 2; i < units[1] + 1; ++i) {
    std::stringstream name;
    name << "stage2_unit" << i;
    unit = ResidualUnit(unit, filter_list[1], 1, true, name.str());
  }

  // res4
  unit = ResidualUnit(unit, filter_list[2], 2, false, "stage3_unit1");
  for (uint32_t i = 2; i < units[2] + 1; ++i) {
    std::stringstream name;
    name << "stage3_unit" << i;
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
    name << "stage4_unit" << i;
    unit = ResidualUnit(unit, filter_list[3], 1, true, name.str());
  }
  auto bn1 = BatchNormUnit(unit, "bn1", false);

  auto relu1 = Activation("relu1", bn1, "relu");

  auto pool1 =
      Pooling("pool1", relu1, Shape(7, 7), PoolingPoolType::kAvg, true);

  return pool1;
}
