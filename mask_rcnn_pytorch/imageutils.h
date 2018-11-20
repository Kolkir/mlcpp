#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H

#include <torch/torch.h>

at::Tensor LoadImage(const std::string path);

#endif  // IMAGEUTILS_H
