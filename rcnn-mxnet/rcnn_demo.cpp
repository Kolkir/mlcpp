#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>

#include "bbox.h"
#include "coco.h"
#include "imageutils.h"
#include "params.h"
#include "rcnn.h"

#include <experimental/filesystem>
#include <iostream>
#include <string>

namespace fs = std::experimental::filesystem;

static mxnet::cpp::Context global_ctx(mxnet::cpp::kCPU, 0);

const cv::String keys =
    "{help h usage ? |      | print this message   }"
    "{@coco_path     |<none>| path to coco dataset }"
    "{@params        |<none>| path to trained parameters }"
    "{@image         |<none>| path to image }";

int main(int argc, char** argv) {
  using namespace mxnet::cpp;
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("CNN test");

  if (parser.has("help") || argc == 1) {
    parser.printMessage();
    return 0;
  }

  std::string coco_path = parser.get<cv::String>(0);
  std::string params_path = parser.get<cv::String>(1);
  std::string image_path = parser.get<cv::String>(2);

  // Chech parsing errors
  if (!parser.check()) {
    parser.printErrors();
    parser.printMessage();
    return 1;
  }

  try {
    coco_path = fs::canonical(fs::absolute(coco_path));
    params_path = fs::canonical(fs::absolute(params_path));
    image_path = fs::canonical(fs::absolute(image_path));
    if (fs::exists(image_path)) {
      std::cout << "Path to the data set : " << coco_path << std::endl;
      std::cout << "Path to the net parameters : " << params_path << std::endl;
      std::cout << "Path to the image : " << image_path << std::endl;

      //----------- Make net
      Params params;
      auto net = GetRCNNSymbol(params, false);

      //----------- Load data
      auto [img, scale] =
          LoadImage(image_path, params.img_short_side, params.img_long_side);
      if (img.empty()) {
        std::cout << "Failed to load image" << std::endl;
        return 1;
      }
      mxnet::cpp::NDArray data(
          mxnet::cpp::Shape(1, 3, static_cast<mxnet::cpp::index_t>(img.rows),
                            static_cast<mxnet::cpp::index_t>(img.cols)),
          global_ctx, false);
      auto array = CVToMxnetFormat(img);
      data.SyncCopyFromCPU(array.data(), array.size());

      mxnet::cpp::NDArray im_info(mxnet::cpp::Shape(1, 3), global_ctx, false);
      std::vector<float> raw_im_info = {static_cast<float>(img.rows),
                                        static_cast<float>(img.cols), scale};
      im_info.SyncCopyFromCPU(raw_im_info.data(), raw_im_info.size());
      mxnet::cpp::NDArray::WaitAll();

      //----------- Load params
      std::map<std::string, mxnet::cpp::NDArray> args_map;
      std::map<std::string, mxnet::cpp::NDArray> aux_map;
      std::tie(args_map, aux_map) = LoadNetParams(global_ctx, params_path);
      args_map["data"] = data;
      args_map["im_info"] = im_info;

      //----------- Check Shapes - shouldn't fail
      std::vector<std::string> args = net.ListArguments();
      std::vector<std::string> outs = net.ListOutputs();
      std::vector<std::string> auxs = net.ListAuxiliaryStates();
      std::map<std::string, std::vector<mx_uint>> arg_shapes;
      for (const auto& arg_name : args) {
        auto iter = args_map.find(arg_name);
        if (iter != args_map.end()) {
          arg_shapes[arg_name] = iter->second.GetShape();
        } else {
          std::cout << "Missed argument : " << arg_name << std::endl;
        }
      }

      for (const auto& arg_name : auxs) {
        auto iter = aux_map.find(arg_name);
        if (iter == aux_map.end()) {
          std::cout << "Missed auxiliary state : " << arg_name << std::endl;
        }
      }

      std::vector<std::vector<mx_uint>> in_shape;
      std::vector<std::vector<mx_uint>> aux_shape;
      std::vector<std::vector<mx_uint>> out_shape;
      net.InferShape(arg_shapes, &in_shape, &aux_shape, &out_shape);

      //----------- Predict
      mxnet::cpp::Executor* executor = net.SimpleBind(
          global_ctx, args_map, std::map<std::string, mxnet::cpp::NDArray>(),
          std::map<std::string, mxnet::cpp::OpReqType>(), aux_map);

      executor->Forward(false);
      auto rois = executor->outputs[0].Copy(Context(kCPU, 0));
      auto scores = executor->outputs[1].Copy(Context(kCPU, 0));
      auto bbox_deltas = executor->outputs[2].Copy(Context(kCPU, 0));
      NDArray::WaitAll();
      delete executor;

      //--------- Decode result
      //      for (size_t i = 0; i < rois.GetShape()[0]; ++i) {
      //        for (size_t j = 0; j < rois.GetShape()[1]; ++j)
      //          std::cout << " " << rois.At(i, j);
      //        std::cout << std::endl;
      //      }

      auto det =
          DecodePredictions(NDArray2ToEigen(rois), NDArray3ToEigen(scores),
                            NDArray3ToEigen(bbox_deltas),
                            Eigen::Map<Eigen::MatrixXf>(
                                raw_im_info.data(), 1,
                                static_cast<Eigen::Index>(raw_im_info.size())),
                            params);

      //-------- Show result
      ShowResult(det, image_path, "det.png", Coco::GetClasses());
    }
    MXNotifyShutdown();
  } catch (const dmlc::Error& err) {
    std::cout << "MXNet error occured : \n";
    auto mx_err_msg = MXGetLastError();
    if (mx_err_msg)
      std::cout << mx_err_msg << "\n";
    else {
      std::cout << err.what() << std::endl;
    }
    return 1;
  } catch (const std::exception& err) {
    std::cout << err.what() << std::endl;
    return 1;
  }
  return 0;
}
