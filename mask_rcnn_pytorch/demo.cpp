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

class InferenceConfig : public Config {
 public:
  InferenceConfig() {
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
//  std::vector<float> blob = {1, 2, 3, 4, 5, 6, 7, 8, 9,
//                             1, 2, 3, 4, 5, 6, 7, 8, 9};
//  auto x = torch::from_blob(blob.data(), {1, 2, 3, 3});

//  std::cerr << x;
//  exit(0);
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
    if (config->gpu_count > 0)
      model->to(torch::DeviceType::CUDA);

    // Load weights trained on MS - COCO
    if (params_path.find(".json") != std::string::npos) {
      torch::autograd::GradMode::set_enabled(
          false);  // make parameters copying possible
      auto new_params = LoadStateDict(params_path);
      auto params = model->named_parameters(true /*recurse*/);
      auto buffers = model->named_buffers(true /*recurse*/);

      for (auto& val : new_params) {
        auto name = val.key();
        // fix naming
        auto pos = name.find("running_var");
        if (pos != std::string::npos) {
          name.replace(pos, 11, "running_variance");
        }

        auto* t = params.find(name);
        if (t != nullptr) {
          std::cout << name << " copy\n";
          t->copy_(val.value());
        } else {
          t = buffers.find(name);
          if (t != nullptr) {
            std::cout << name << " copy\n";
            t->copy_(val.value());
          } else {
            // throw std::logic_error(name + " parameter not found!");
            std::cout << name + " parameter not found!\n";
          }
        }
      }
      torch::autograd::GradMode::set_enabled(true);

      // torch::save(model, "params.dat");
      std::cout << "Model state converted!\n";
      // exit(0);
    } else {
      torch::load(model, params_path);
    }
    std::cout.flush();

    auto [detections, mrcnn_mask] = model->Detect(molded_images, image_metas);

    // Process detections
    //[final_rois, final_class_ids, final_scores, final_masks]
    using Result =
        std::tuple<at::Tensor, at::Tensor, at::Tensor, std::vector<cv::Mat>>;
    std::vector<Result> results;
    for (size_t i = 0; i < images.size(); ++i) {
      auto result = UnmoldDetections(detections[static_cast<int64_t>(i)],
                                     mrcnn_mask[static_cast<int64_t>(i)],
                                     image.size(), windows[i]);
      results.push_back(result);
    }

    visualize(image, std::get<0>(results[0]), std::get<1>(results[0]),
              std::get<2>(results[0]), std::get<3>(results[0]));

  } catch (const std::exception& err) {
    std::cout << err.what() << std::endl;
    return 1;
  }
  return 0;
}
