#ifndef RCNN_H
#define RCNN_H

#include "params.h"

#include <mxnet-cpp/MxNetCpp.h>

mxnet::cpp::Symbol GetRCNNSymbol(const Params& params);

#endif  // RCNN_H
