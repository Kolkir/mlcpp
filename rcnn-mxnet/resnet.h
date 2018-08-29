#ifndef RESNET_H
#define RESNET_H

#include "params.h"

#include <mxnet-cpp/MxNetCpp.h>

mxnet::cpp::Symbol GetResnetHeadSymbol(
    mxnet::cpp::Symbol data,
    const std::vector<uint32_t>& units,
    const std::vector<uint32_t>& filter_list);
mxnet::cpp::Symbol GetResnetTopSymbol(mxnet::cpp::Symbol data,
                                      const std::vector<uint32_t>& units,
                                      const std::vector<uint32_t>& filter_list);

#endif  // RESNET_H
