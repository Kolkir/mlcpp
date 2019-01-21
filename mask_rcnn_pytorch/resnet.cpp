#include "resnet.h"
#include "debug.h"

ResNetImpl::ResNetImpl() {}

ResNetImpl::ResNetImpl(Architecture architecture, bool stage5)
    : stage5_{stage5} {
  assert(architecture == Architecture::ResNet50 ||
         architecture == Architecture::ResNet101);

  if (architecture == Architecture::ResNet50) {
    layers_[2] = 6;
  } else if (architecture == Architecture::ResNet101) {
    layers_[2] = 23;
  }

  c1_ = torch::nn::Sequential(
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3)),
      torch::nn::BatchNorm(
          torch::nn::BatchNormOptions(64).eps(0.001).momentum(0.01)),
      torch::nn::Functional(torch::relu),
      SamePad2d(/*kernel_size*/ 3, /*stride*/ 2),
      torch::nn::Functional(torch::max_pool2d,
                            /*kernel_size*/ at::IntList({3, 3}),
                            /*stride*/ at::IntList({2, 2}),
                            /*padding*/ at::IntList({0, 0}),
                            /*dilation*/ at::IntList({1, 1}),
                            /*ceil_mode*/ false));
  this->register_module("C1", c1_);

  c2_ = MakeLayer(64, layers_[0]);
  this->register_module("C2", c2_);

  c3_ = MakeLayer(128, layers_[1], /*stride*/ 2);
  this->register_module("C3", c3_);

  c4_ = MakeLayer(256, layers_[2], /*stride*/ 2);
  this->register_module("C4", c4_);

  if (stage5_) {
    c5_ = MakeLayer(512, layers_[3], /*stride*/ 2);
    this->register_module("C5", c5_);
  }
}

torch::nn::Sequential ResNetImpl::MakeLayer(uint32_t planes,
                                            uint32_t blocks,
                                            uint32_t stride) {
  torch::nn::Sequential downsample;
  if (stride != 1 || inplanes_ != planes * BottleneckImpl::expansion) {
    downsample = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(
                              inplanes_, planes * BottleneckImpl::expansion, 1)
                              .stride(stride)),
        torch::nn::BatchNorm(
            torch::nn::BatchNormOptions(planes * BottleneckImpl::expansion)
                .eps(0.001)
                .momentum(0.01)));
  }
  torch::nn::Sequential layers(
      Bottleneck(inplanes_, planes, stride, downsample));
  inplanes_ = planes * BottleneckImpl::expansion;
  for (uint32_t i = 1; i < blocks; ++i) {
    layers->push_back(Bottleneck(inplanes_, planes));
  }

  return layers;
}

at::Tensor ResNetImpl::forward(at::Tensor input) {
  input = c1_->forward(input);
  input = c2_->forward(input);
  input = c3_->forward(input);
  input = c4_->forward(input);
  if (c5_)
    input = c5_->forward(input);
  return input;
}

BottleneckImpl::BottleneckImpl() {}

BottleneckImpl::BottleneckImpl(uint32_t inplanes,
                               uint32_t planes,
                               uint32_t stride,
                               torch::nn::Sequential downsample)
    : conv1_{torch::nn::Conv2dOptions(inplanes, planes, 1).stride(stride)},
      bn1_{torch::nn::BatchNormOptions(planes).eps(0.001).momentum(0.01)},
      padding2_(/*kernel_size*/ 3, /*stride*/ 1),
      conv2_{torch::nn::Conv2dOptions(planes, planes, 3)},
      bn2_{torch::nn::BatchNormOptions(planes).eps(0.001).momentum(0.01)},
      conv3_{torch::nn::Conv2dOptions(planes, planes * 4, 1)},
      bn3_{torch::nn::BatchNormOptions(planes * 4).eps(0.001).momentum(0.01)},
      relu_{torch::relu},
      downsample_{downsample} {
  register_module("conv1", conv1_);
  register_module("bn1", bn1_);
  register_module("padding2", padding2_);
  register_module("conv2", conv2_);
  register_module("bn2", bn2_);
  register_module("conv3", conv3_);
  register_module("bn3", bn3_);
  register_module("relu", relu_);
  if (downsample_)
    register_module("downsample", downsample_);
}

at::Tensor BottleneckImpl::forward(at::Tensor x) {
  auto residual = x;

  at::Tensor out = conv1_->forward(x);
  out = bn1_->forward(out);
  out = relu_->forward(out);

  out = padding2_->forward(out);
  out = conv2_->forward(out);
  out = bn2_->forward(out);
  out = relu_->forward(out);

  out = conv3_->forward(out);
  out = bn3_->forward(out);

  if (downsample_)
    residual = downsample_->forward(x);

  out += residual;
  out = relu_->forward(out);

  return out;
}
