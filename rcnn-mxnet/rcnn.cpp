#include "coco.h"
#include "trainiter.h"

#include <opencv2/opencv.hpp>

#include <experimental/filesystem>
#include <iostream>
#include <string>

namespace fs = std::experimental::filesystem;

/*The global context, change them if necessary*/
static mxnet::cpp::Context global_ctx(mxnet::cpp::kGPU, 0);
// static Context global_ctx(mxnet::cpp::kCPU,0);

const cv::String keys =
    "{help h usage ? |      | print this message   }"
    "{@coco_path     |<none>| path to coco dataset }";

int main(int argc, char** argv) {
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Faster R-CNN");

  if (parser.has("help") || argc == 1) {
    parser.printMessage();
    return 0;
  }

  std::string coco_path = parser.get<cv::String>(0);

  // Chech parsing errors
  if (!parser.check()) {
    parser.printErrors();
    return 1;
  }

  try {
    coco_path = fs::canonical(fs::absolute(coco_path));
    if (fs::exists(coco_path)) {
      std::cout << "Path to the data set : " << coco_path << std::endl;
      Coco coco(coco_path);
      coco.LoadTrainData();
      TrainIter data_iter(global_ctx, &coco, 8);

    } else {
      std::cout << "Ivalid path to the data set : " << coco_path << std::endl;
      return 1;
    }
  } catch (const std::exception& err) {
    std::cout << err.what() << std::endl;
    return 1;
  }

  return 0;
}
