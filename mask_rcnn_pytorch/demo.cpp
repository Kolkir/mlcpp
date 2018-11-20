#include "config.h"
#include "imageutils.h"
#include "maskrcnn.h"

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <experimental/filesystem>
#include <iostream>
#include <memory>

namespace fs = std::experimental::filesystem;

class InferenceConfig : public Config {
 public:
  InferenceConfig() {
    if (!torch::cuda::is_available())
      throw std::runtime_error("Cuda is not available");
    gpu_count = 1;
    images_per_gpu = 1;
    UpdateSettings();
  }
};

const cv::String keys =
    "{help h usage ? |      | print this message   }"
    "{@params        |<none>| path to trained parameters }"
    "{@image         |<none>| path to image }";

int main(int argc, char** argv) {
  try {
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MaskRCNN demo");

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

    // Root directory of the project
    auto root_dir = fs::current_path();
    // Directory to save logs and trained model
    auto model_dir = root_dir / "logs";

    auto config = std::make_shared<InferenceConfig>();

    // Create model object.
    MaskRCNN model(model_dir, config);
    if (config->gpu_count > 0)
      model.to(torch::DeviceType::CUDA);

    // Load weights trained on MS-COCO
    torch::serialize::InputArchive archive;
    archive.load_from(params_path);
    model.load(archive);

    auto image = LoadImage(image_path);
    auto results = model.detect(image);
  } catch (const std::exception& err) {
    std::cout << err.what() << std::endl;
    return 1;
  }
  return 0;
}
