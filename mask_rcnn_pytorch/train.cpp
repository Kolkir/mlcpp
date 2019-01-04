#include "cocoloader.h"
#include "config.h"
#include "debug.h"
#include "imageutils.h"
#include "maskrcnn.h"
#include "stateloader.h"
#include "visualize.h"

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
    gpu_count = 0;
    images_per_gpu = 1;
    num_classes = 81;  // for coco dataset
    UpdateSettings();
  }
};

const cv::String keys =
    "{help h usage ? |      | print this message   }"
    "{@params        |<none>| path to trained parameters }"
    "{@image         |<none>| path to image }";

int main(int argc, char** argv) {
  CocoLoader cloader(
      "/media/disk2/data_sets/coco/train2017",
      "/media/disk2/data_sets/coco/annotations/instances_train2017.json");
  cloader.LoadData();
  auto im_desc = cloader.GetImage(4566);
  std::cerr << im_desc.id << "\n";
  cv::Mat img = cloader.DrawAnnotedImage(im_desc.id);
  cv::imwrite("coco_test.png", img);
  exit(0);
#ifndef NDEBUG
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

    std::string params_path = parser.get<cv::String>(0);
    std::string image_path = parser.get<cv::String>(1);

    // Chech parsing errors
    if (!parser.check()) {
      parser.printErrors();
      parser.printMessage();
      return 1;
    }

    params_path = fs::canonical(params_path);
    if (!fs::exists(params_path))
      throw std::invalid_argument("Wrong file path for parameters");

    image_path = fs::canonical(image_path);
    if (!fs::exists(image_path))
      throw std::invalid_argument("Wrong file path forimage");

    auto config = std::make_shared<TrainConfig>();

    // Root directory of the project
    auto root_dir = fs::current_path();
    // Directory to save logs and trained model
    auto model_dir = root_dir / "logs";

    // Create model object.
    MaskRCNN model(model_dir, config);
    if (config->gpu_count > 0)
      model->to(torch::DeviceType::CUDA);

  } catch (const std::exception& err) {
    std::cout << err.what() << std::endl;
    return 1;
  }
  return 0;
}
