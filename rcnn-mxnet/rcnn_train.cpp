#include "coco.h"
#include "imageutils.h"
#include "metrics.h"
#include "params.h"
#include "rcnn.h"
#include "trainiter.h"

#define NCURSES_OUT

#ifdef NCURSES_OUT
#include <ncurses.h>
#endif

#include <opencv2/opencv.hpp>

#include <experimental/filesystem>
#include <iostream>
#include <string>

namespace fs = std::experimental::filesystem;

static mxnet::cpp::Context global_ctx(mxnet::cpp::kGPU, 0);
// static mxnet::cpp::Context global_ctx(mxnet::cpp::kCPU, 0);

const cv::String keys =
    "{help h usage ? |                  | print this message   }"
    "{@coco_path     |<none>            | path to coco dataset }"
    "{p params       |                  | path to trained parameters }"
    "{s start-train  |                  | path to trained parameters }"
    "{c check-point  |check-point.params| check point file name }";

int main(int argc, char** argv) {
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Faster R-CNN demo");

  if (argc == 1) {
    if (parser.has("help"))
      parser.printMessage();
    return 1;
  }

  std::string coco_path = parser.get<cv::String>(0);
  std::string params_path;
  if (parser.has("params"))
    params_path = parser.get<cv::String>("params");

  std::string check_point_file = parser.get<cv::String>("check-point");

  bool start_train{false};
  if (parser.has("start-train"))
    start_train = true;

  // Chech parsing errors
  if (!parser.check()) {
    parser.printErrors();
    if (parser.has("help"))
      parser.printMessage();
    return 1;
  }

  try {
    check_point_file = fs::canonical(fs::absolute(check_point_file));
    coco_path = fs::canonical(fs::absolute(coco_path));
    if (!params_path.empty())
      params_path = fs::canonical(fs::absolute(params_path));
    if (fs::exists(coco_path)) {
      std::cout << "Path to the data set : " << coco_path << std::endl;
      std::cout << "Path to the net parameters : " << params_path << std::endl;
      std::cout << "Path to the check-point file : " << check_point_file
                << std::endl;

      Params params;
      auto net = GetRCNNSymbol(params, true);

      std::map<std::string, mxnet::cpp::NDArray> args_map;
      std::map<std::string, mxnet::cpp::NDArray> aux_map;
      if (!params_path.empty())
        std::tie(args_map, aux_map) = LoadNetParams(global_ctx, params_path);

      // ---------- Test parametes & Check Shapes - shouldn't fail
      std::map<std::string, std::vector<mx_uint>> arg_shapes;
      arg_shapes["data"] = {params.rcnn_batch_size, 3, params.img_short_side,
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
      arg_shapes["gt_boxes"] = {params.rcnn_batch_size,
                                params.rcnn_batch_gt_boxes, 5};
      arg_shapes["label"] = {params.rcnn_batch_size, 1,
                             rpn_num_anchors * feat_height, feat_width};
      arg_shapes["bbox_target"] = {params.rcnn_batch_size, 4 * rpn_num_anchors,
                                   feat_height, feat_width};
      arg_shapes["bbox_weight"] = {params.rcnn_batch_size, 4 * rpn_num_anchors,
                                   feat_height, feat_width};

      std::vector<std::string> args = net.ListArguments();
      if (!params_path.empty()) {
        std::vector<std::string> outs = net.ListOutputs();
        std::vector<std::string> auxs = net.ListAuxiliaryStates();
        for (const auto& arg_name : args) {
          auto iter = args_map.find(arg_name);
          if (iter != args_map.end()) {
            arg_shapes[arg_name] = iter->second.GetShape();
          } else {
            std::cout << "Configurable or Missed argument : " << arg_name
                      << std::endl;
          }
        }

        for (const auto& arg_name : auxs) {
          auto iter = aux_map.find(arg_name);
          if (iter != aux_map.end()) {
          } else {
            std::cout << "Missed auxiliary state : " << arg_name << std::endl;
          }
        }
      }
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

      if (start_train) {
        net.InferArgsMap(global_ctx, &args_map, args_map);
      }

      //----------- Train
      uint32_t max_epoch = 10;
      float learning_rate = 1e-4f;
      float weight_decay = 1e-4f;

      mxnet::cpp::Executor* executor{nullptr};
      // without aux_map - training fails with nans
      executor = net.SimpleBind(
          global_ctx, args_map, std::map<std::string, mxnet::cpp::NDArray>(),
          std::map<std::string, mxnet::cpp::OpReqType>(), aux_map);

      if (start_train) {
        InitiaizeRCNN(args_map);
      }

      std::cout << "Loading trainig data set ..." << std::endl;
      Coco coco(coco_path);
      coco.LoadTrainData();
      TrainIter train_iter(global_ctx, &coco, params);
      auto batch_count = train_iter.GetBatchCount();
      std::cout << "Total images count: " << train_iter.GetSize() << std::endl;
      std::cout << "Batch count: " << batch_count << std::endl;

      mxnet::cpp::Optimizer* opt = mxnet::cpp::OptimizerRegistry::Find("ccsgd");
      opt->SetParam("lr", learning_rate)
          ->SetParam("wd", weight_decay)
          ->SetParam("momentum", 0.9)
          ->SetParam("rescale_grad", 1.0 / params.rcnn_batch_size)
          ->SetParam("clip_gradient", 5);

      std::unordered_set<std::string> not_update_args{
          "data", "im_info", "gt_boxes", "label", "bbox_target", "bbox_weight"};
      for (auto& arg_name : args) {
        if (arg_name.find("conv0") != std::string::npos ||
            arg_name.find("stage1") != std::string::npos ||
            arg_name.find("gamma") != std::string::npos ||
            arg_name.find("beta") != std::string::npos) {
          not_update_args.insert(arg_name);
        }
      }

#ifdef NCURSES_OUT
      initscr();
#endif
      RCNNAccMetric acc_metric;
      RCNNLogLossMetric log_loss_metric;
      uint32_t batch_num = 0;
      for (uint32_t epoch = 0; epoch < max_epoch; ++epoch) {
#ifdef NCURSES_OUT
        erase();
        mvprintw(0, 0, "Epoch: %d\n", epoch);
#else
        std::cout << "Epoch: " << epoch << std::endl;
#endif
        train_iter.Reset();
        while (train_iter.Next(feat_height, feat_width)) {
#ifdef NCURSES_OUT
          mvprintw(1, 0, "Batch: %d \\ %d\n", batch_count, batch_num);
#else
          std::cout << "Batch: " << batch_num << std::endl;
#endif
          train_iter.GetImData(args_map["data"]);
          train_iter.GetImInfoData(args_map["im_info"]);
          train_iter.GetGtBoxesData(args_map["gt_boxes"]);
          train_iter.GetLabel(args_map["label"]);
          train_iter.GetBBoxTraget(args_map["bbox_target"]);
          train_iter.GetBBoxWeight(args_map["bbox_weight"]);
          mxnet::cpp::NDArray::WaitAll();
#ifdef NCURSES_OUT
          mvprintw(2, 0, "Batch data filled\n");
#else
          std::cout << "Batch data filled" << std::endl;
#endif

          executor->Forward(true);
          executor->Backward();

          for (size_t i = 0; i < args.size(); ++i) {
            if (not_update_args.find(args[i]) != not_update_args.end())
              continue;
            executor->grad_arrays[i].WaitToRead();
            executor->arg_arrays[i].WaitToWrite();
            opt->Update(static_cast<int>(i), executor->arg_arrays[i],
                        executor->grad_arrays[i]);
            executor->arg_arrays[i].WaitToRead();
          }
#ifdef NCURSES_OUT
          mvprintw(3, 0, "Parameters updated\n");
#else
          std::cout << "Parameters updated\n" << std::endl;
#endif
          // evaluate metrics
          executor->Forward(false);
          executor->outputs[4].WaitToRead();
          executor->outputs[2].WaitToRead();
          acc_metric.Update(executor->outputs[4], executor->outputs[2]);
          log_loss_metric.Update(executor->outputs[4], executor->outputs[2]);
#ifdef NCURSES_OUT
          mvprintw(4, 0, "Batch RCNN accurary: %f\n",
                   static_cast<double>(acc_metric.Get()));
          mvprintw(5, 0, "Batch RCNN log loss %f\n",
                   static_cast<double>(log_loss_metric.Get()));
          refresh();
#else
          std::cout << "Batch RCNN accurary " << acc_metric.Get() << std::endl;
          std::cout << "Batch RCNN log loss " << log_loss_metric.Get()
                    << std::endl;
#endif
          ++batch_num;
        }
        SaveNetParams(check_point_file, executor);
      }
#ifdef NCURSES_OUT
      endwin();
#endif

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
