#include "cocodataset.h"
#include "cocoloader.h"
#include "config.h"
#include "debug.h"
#include "imageutils.h"
#include "maskrcnn.h"
#include "stateloader.h"

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <experimental/filesystem>
#include <iostream>
#include <memory>

namespace fs = std::experimental::filesystem;

class TrainConfig : public Config {
 public:
  TrainConfig() {
    if (!torch::cuda::is_available())
      throw std::runtime_error("Cuda is not available");
    gpu_count = 1;
    images_per_gpu = 1;
    num_classes = 81;  // 4 - for shapes, 81 - for coco dataset

    UpdateSettings();
  }
};

const cv::String keys =
    "{help h usage ? |      | print this message   }"
    "{@data_dir      |<none>| path to coco dataset root folder}"
    "{@params        |<none>| path to trained parameters }";

int main(int argc, char** argv) {
#ifndef NDEBUG
  at::globalContext().setDeterministicCuDNN(true);
  torch::manual_seed(9993);

  // initialize debug print function
  auto x__ = torch::tensor({1, 2, 3, 4});
  auto p = PrintTensor(x__);
#endif
  try {
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MaskRCNN train");

    if (parser.has("help") || argc == 1) {
      parser.printMessage();
      return 0;
    }

    std::string data_path = parser.get<cv::String>(0);
    std::string params_path = parser.get<cv::String>(1);

    // Chech parsing errors
    if (!parser.check()) {
      parser.printErrors();
      parser.printMessage();
      return 1;
    }

    params_path = fs::canonical(params_path);
    if (!fs::exists(params_path))
      throw std::invalid_argument("Wrong file path for parameters");

    auto config = std::make_shared<TrainConfig>();

    // Root directory of the project
    auto root_dir = fs::current_path();
    // Directory to save logs and trained model
    auto model_dir = root_dir / "logs";
    if (!fs::exists(model_dir)) {
      fs::create_directories(model_dir);
    }

    // Create model object.
    MaskRCNN model(model_dir, config);

    // load weights before moving to GPU
    if (params_path.find(".json") != std::string::npos) {
      LoadStateDictJson(*model, params_path);
    } else {
      // Uncoment to load only resnet
      //      std::string ignore_layers =
      //          "(fpn.P5\\_.*)|(fpn.P4\\_.*)|(fpn.P3\\_.*)|(fpn.P2\\_.*)|(rpn.*)|("
      //          "classifier.*)|(mask.*)";
      std::string ignore_layers{""};
      LoadStateDict(*model, params_path, ignore_layers);
    }

    if (config->gpu_count > 0)
      model->to(torch::DeviceType::CUDA);

    // Make data sets
    auto train_loader = std::make_unique<CocoLoader>(
        fs::path(data_path) / "train2017",
        fs::path(data_path) / "annotations/instances_train2017.json");
    auto train_set =
        std::make_unique<CocoDataset>(std::move(train_loader), config);

    auto val_loader = std::make_unique<CocoLoader>(
        fs::path(data_path) / "val2017",
        fs::path(data_path) / "annotations/instances_val2017.json");
    auto val_set = std::make_unique<CocoDataset>(std::move(val_loader), config);

    //    // Training - Stage 1
    std::cout << "Training network heads" << std::endl;
    model->Train(*train_set, *val_set, config->learning_rate, /*epochs*/
                 40,
                 "heads");  // 40

    //    // Training - Stage 2
    std::cout << "Fine tune Resnet stage 4 and up" << std::endl;
    model->Train(*train_set, *val_set, config->learning_rate, /*epochs*/
                 120,
                 "4+");  // 120

    // Training - Stage 3
    // Fine tune all layers
    std::cout << "Fine tune all layers" << std::endl;
    model->Train(*train_set, *val_set, config->learning_rate / 10,
                 /*epochs*/ 160, "all");  // 160

  } catch (const std::exception& err) {
    std::cout << err.what() << std::endl;
    return 1;
  }
  return 0;
}
