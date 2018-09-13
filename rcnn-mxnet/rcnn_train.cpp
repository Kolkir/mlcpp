#include "coco.h"
#include "imageutils.h"
#include "params.h"
#include "rcnn.h"
#include "trainiter.h"

#include <opencv2/opencv.hpp>

#include <experimental/filesystem>
#include <iostream>
#include <string>

namespace fs = std::experimental::filesystem;

// static mxnet::cpp::Context global_ctx(mxnet::cpp::kGPU, 0);
static mxnet::cpp::Context global_ctx(mxnet::cpp::kCPU, 0);

const cv::String keys =
    "{help h usage ? |      | print this message   }"
    "{@coco_path     |<none>| path to coco dataset }"
    "{@params        |<none>| path to trained parameters }"
    "{@image         |<none>| path to image }";

int main(int argc, char** argv) {
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Faster R-CNN demo");

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
    if (fs::exists(coco_path)) {
      std::cout << "Path to the data set : " << coco_path << std::endl;
      std::cout << "Path to the net parameters : " << params_path << std::endl;
      std::cout << "Path to the image : " << image_path << std::endl;
      // Coco coco(coco_path);
      // coco.LoadTrainData();
      Params params;
      auto net = GetRCNNSymbol(params, true);

      std::map<std::string, mxnet::cpp::NDArray> args_map;
      std::map<std::string, mxnet::cpp::NDArray> aux_map;
      // std::tie(args_map, aux_map) = LoadNetParams(global_ctx, params_path);

      // ---------- Test parametes & Check Shapes - shouldn't fail
      std::map<std::string, std::vector<mx_uint>> arg_shapes;
      arg_shapes["data"] = {params.rcnn_batch_size, 3, params.img_long_side,
                            params.img_long_side};

      auto feat_sym = net.GetInternals()["rpn_cls_score_output"];

      std::vector<std::vector<mx_uint>> in_shape;
      std::vector<std::vector<mx_uint>> aux_shape;
      std::vector<std::vector<mx_uint>> out_shape;
      feat_sym.InferShape(arg_shapes, &in_shape, &aux_shape, &out_shape);
      mx_uint feat_height = out_shape.at(0).at(2);
      mx_uint feat_width = out_shape.at(0).at(3);
      mx_uint rpn_num_anchors = static_cast<mx_uint>(
          params.rpn_anchor_scales.size() * params.rpn_anchor_ratios.size());

      arg_shapes["im_info"] = {params.rcnn_batch_size, 3};
      arg_shapes["gt_boxes"] = {params.rcnn_batch_size, 100, 5};
      arg_shapes["label"] = {params.rcnn_batch_size, 1,
                             rpn_num_anchors * feat_height, feat_width};
      arg_shapes["bbox_target"] = {params.rcnn_batch_size, 4 * rpn_num_anchors,
                                   feat_height, feat_width};
      arg_shapes["bbox_weight"] = {params.rcnn_batch_size, 4 * rpn_num_anchors,
                                   feat_height, feat_width};

      //      std::vector<std::string> args = net.ListArguments();
      //      std::vector<std::string> outs = net.ListOutputs();
      //      std::vector<std::string> auxs = net.ListAuxiliaryStates();
      //      for (const auto& arg_name : args) {
      //        auto iter = args_map.find(arg_name);
      //        if (iter != args_map.end()) {
      //          arg_shapes[arg_name] = iter->second.GetShape();
      //        } else {
      //          std::cout << "Missed argument : " << arg_name << std::endl;
      //        }
      //      }

      //      for (const auto& arg_name : auxs) {
      //        auto iter = aux_map.find(arg_name);
      //        if (iter != aux_map.end()) {
      //        } else {
      //          std::cout << "Missed auxiliary state : " << arg_name <<
      //          std::endl;
      //        }
      //      }
      in_shape.clear();
      aux_shape.clear();
      out_shape.clear();
      net.InferShape(arg_shapes, &in_shape, &aux_shape, &out_shape);

      //----------- Initialize binding arrays

      // train inputs
      args_map["data"] = mxnet::cpp::NDArray(
          mxnet::cpp::Shape(arg_shapes["data"]), global_ctx, false);
      args_map["im_info"] = mxnet::cpp::NDArray(
          mxnet::cpp::Shape(arg_shapes["im_info"]), global_ctx, false);
      args_map["gt_boxes"] = mxnet::cpp::NDArray(
          mxnet::cpp::Shape(arg_shapes["gt_boxes"]), global_ctx, false);

      // train outputs
      args_map["label"] = mxnet::cpp::NDArray(
          mxnet::cpp::Shape(arg_shapes["label"]), global_ctx, false);
      args_map["bbox_target"] = mxnet::cpp::NDArray(
          mxnet::cpp::Shape(arg_shapes["bbox_target"]), global_ctx, false);
      args_map["bbox_weight"] = mxnet::cpp::NDArray(
          mxnet::cpp::Shape(arg_shapes["bbox_weight"]), global_ctx, false);

      net.InferArgsMap(global_ctx, &args_map, args_map);

      //----------- Predict
      mxnet::cpp::Executor* executor = net.SimpleBind(global_ctx, args_map);

      executor->Forward(true);
      mxnet::cpp::NDArray::WaitAll();
      delete executor;

      MXNotifyShutdown();
    } else {
      std::cout << "Ivalid path to the data set : " << coco_path << std::endl;
      return 1;
    }
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
