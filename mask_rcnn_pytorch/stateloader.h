#ifndef STATELOADER_H
#define STATELOADER_H

#include <torch/torch.h>
#include <vector>

/* Correspondig Python export code
 * raw_state_dict = {}
 * for k, v in model.state_dict().items():
 *     if isinstance(v, torch.Tensor):
 *         raw_state_dict[k] = (list(v.size()), v.numpy().tolist())
 *         break
 *     else:
 *         print("State parameter type error : {}".format(k))
 *         exit(-1)
 *
 * with open('mask_rcnn_coco.json', 'w') as outfile:
 *     json.dump(raw_state_dict, outfile)
 */

torch::OrderedDict<std::string, torch::Tensor> LoadStateDict(
    const std::string& file_name);

void LoadStateDict(torch::nn::Module& module, const std::string& file_name);

#endif  // STATELOADER_H
