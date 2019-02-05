#include "cocoloader.h"
#include "config.h"
#include "datasetclasses.h"
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

class InferenceConfig : public Config {
 public:
  InferenceConfig() {
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
    "{@params        |<none>| path to trained parameters }"
    "{@image         |<none>| path to image }";

int main(int argc, char** argv) {
#ifndef NDEBUG
  // initialize debug print function
  auto x__ = torch::tensor({1, 2, 3, 4});
  auto p = PrintTensor(x__);
#endif
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

    params_path = fs::canonical(params_path);
    if (!fs::exists(params_path))
      throw std::invalid_argument("Wrong file path for parameters");

    image_path = fs::canonical(image_path);
    if (!fs::exists(image_path))
      throw std::invalid_argument("Wrong file path forimage");

    auto config = std::make_shared<InferenceConfig>();

    // Load image
    auto image = LoadImage(image_path);

    std::vector<cv::Mat> images{image};
    // Mold inputs to format expected by the neural network
    auto [molded_images, image_metas, windows] =
        MoldInputs(images, *config.get());

    // Root directory of the project
    auto root_dir = fs::current_path();
    // Directory to save logs and trained model
    auto model_dir = root_dir / "logs";

    // Create model object.
    MaskRCNN model(model_dir, config);

    // load state before moving to GPU
    if (params_path.find(".json") != std::string::npos) {
      LoadStateDictJson(*model, params_path);
    } else {
      LoadStateDict(*model, params_path, "");
    }

    if (config->gpu_count > 0)
      model->to(torch::DeviceType::CUDA);

    auto start = std::chrono::steady_clock::now();
    auto [detections, mrcnn_mask] = model->Detect(molded_images, image_metas);
    if (!is_empty(detections)) {
      // Process detections
      //[final_rois, final_class_ids, final_scores, final_masks]
      using Result =
          std::tuple<at::Tensor, at::Tensor, at::Tensor, std::vector<cv::Mat>>;
      std::vector<Result> results;

      double mask_threshold = 0.5;
      for (size_t i = 0; i < images.size(); ++i) {
        auto result =
            UnmoldDetections(detections[static_cast<int64_t>(i)],
                             mrcnn_mask[static_cast<int64_t>(i)], image.size(),
                             windows[i], mask_threshold);
        results.push_back(result);
      }
      auto stop = std::chrono::steady_clock::now();
      auto inference_time =
          std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
              .count();
      std::cout << "Inference time " << inference_time << "\n";

      float score_threshold = 0.7f;
      visualize(image, std::get<0>(results[0]), std::get<1>(results[0]),
                std::get<2>(results[0]), std::get<3>(results[0]),
                score_threshold, GetDatasetClasses());
    } else {
      std::cerr << "Failed to detect anything!\n";
    }

  } catch (const std::exception& err) {
    std::cout << err.what() << std::endl;
    return 1;
  }
  return 0;
}
