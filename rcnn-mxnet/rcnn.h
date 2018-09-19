#ifndef RCNN_H
#define RCNN_H

#include "params.h"

#include <mxnet-cpp/MxNetCpp.h>

mxnet::cpp::Symbol GetRCNNSymbol(const Params& params, bool train);

void InitiaizeRCNN(std::map<std::string, mxnet::cpp::NDArray>& args_map);

#endif  // RCNN_H
