#ifndef STATELOADER_H
#define STATELOADER_H

#include <torch/torch.h>
#include <vector>

torch::OrderedDict<std::string, torch::Tensor> LoadStateDict(
    const std::string& file_name);

#endif  // STATELOADER_H
