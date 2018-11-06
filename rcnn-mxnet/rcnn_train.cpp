#include "coco.h"
#include "gputrainiter.h"
#include "imageutils.h"
#include "metrics.h"
#include "mxutils.h"
#include "params.h"
#include "rcnn.h"
#include "reporter.h"

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
    "{p params       |                  | path to trained resnet parameters }"
    "{s start-train  |                  | flag to start initial training }"
    "{c check-point  |check-point.params| check point file name }";

int main(int argc, char** argv) {
  MXRandomSeed(5675317);

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
    check_point_file = fs::absolute(check_point_file);
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
      std::cout << "Loading parameters ... " << std::endl;
      if (!params_path.empty())
        std::tie(args_map, aux_map) = LoadNetParams(global_ctx, params_path);

      // ---------- Test parametes & Check Shapes - shouldn't fail
      std::cout << "Test shapes ..." << std::endl;
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
      CheckMXnetError("InferShape");

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
      uint32_t max_epoch = 100;

      mxnet::cpp::Executor* executor{nullptr};
      // without aux_map - training fails with nans
      executor = net.SimpleBind(
          global_ctx, args_map, std::map<std::string, mxnet::cpp::NDArray>(),
          std::map<std::string, mxnet::cpp::OpReqType>(), aux_map);
      args = net.ListArguments();

      //      mxnet::cpp::Monitor monitor(1, std::regex("stage3.*"));
      //      //.*|bbox_pred.*|prop_target.*|bbox_loss.*|rpn.*"));
      //      //
      //      std::regex("data|im_info|gt_boxes|label|bbox_target|bbox_weight"));
      //      monitor.install(executor);

      if (start_train) {
        InitiaizeRCNN(args_map);
        CheckMXnetError("initialize rcnn");
      }

      std::cout << "Indexing trainig data set ..." << std::endl;
      Coco coco(coco_path);
      coco.LoadTrainData(
          {2, 3, 4, 6, 7},  // train only on vehicles with fixed aspect ratio
          static_cast<float>(params.img_long_side) /
              static_cast<float>(params.img_short_side));
      GpuTrainIter train_iter(&coco, params, feat_height, feat_width);
      train_iter.AllocateGpuCache(
          global_ctx, arg_shapes["data"], arg_shapes["im_info"],
          arg_shapes["gt_boxes"], arg_shapes["label"],
          arg_shapes["bbox_target"], arg_shapes["bbox_weight"]);

      auto batch_count = train_iter.GetBatchCount();
      std::cout << "Total images count: " << train_iter.GetSize() << std::endl;
      std::cout << "Batch count: " << batch_count << std::endl;

      float lr = 0.001f;
      float lr_factor = 0.1f;
      int lr_epoch = 10;  // epoch to decay lr
      int lr_step = (lr_epoch * static_cast<int>(train_iter.GetSize())) /
                    static_cast<int>(params.rcnn_batch_size);

      mxnet::cpp::Optimizer* opt = mxnet::cpp::OptimizerRegistry::Find("sgd");
      opt->SetLRScheduler(
             std::make_unique<mxnet::cpp::FactorScheduler>(lr_step, lr_factor))
          ->SetParam("lr", lr)
          ->SetParam("wd", 0.0005)
          ->SetParam("momentum", 0.9f)
          ->SetParam("rescale_grad", 1.0f / params.rcnn_batch_size)
          ->SetParam("clip_gradient", 5);
      CheckMXnetError("create optimizer");

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
#ifdef NDEBUG
      Reporter reporter(false, 8, std::chrono::milliseconds(5000));
#else
      Reporter reporter(true, 8, std::chrono::milliseconds(5000));
#endif
      reporter.SetLineDescription(0,
                                  "Epoch(" + std::to_string(max_epoch) + ")");
      reporter.SetLineDescription(1,
                                  "Batch(" + std::to_string(batch_count) + ")");
      reporter.SetLineDescription(2, "PRN accuracy");
      reporter.SetLineDescription(3, "PRN log loss");
      reporter.SetLineDescription(4, "PRN l1 loss");
      reporter.SetLineDescription(5, "RCNN accurary");
      reporter.SetLineDescription(6, "RCNN log loss");
      reporter.SetLineDescription(7, "RCNN l1 loss");
      reporter.Start();

      RPNAccMetric rpn_acc_metric;
      RPNLogLossMetric rpn_log_loss_metric;
      RPNL1LossMetric rpn_l1_loss_metric;

      RCNNAccMetric rcnn_acc_metric;
      RCNNLogLossMetric rcnn_log_loss_metric;
      RCNNL1LossMetric rcnn_l1_loss_metric;

      uint32_t batch_num = 0;
      for (uint32_t epoch = 0; epoch < max_epoch; ++epoch) {
        batch_num = 0;
        reporter.SetLineValue(0, epoch);
        train_iter.Reset();
        while (train_iter.Next()) {
          reporter.SetLineValue(1, batch_num);
          CheckMXnetError("before copy batch");
          train_iter.GetData(args_map["data"], args_map["im_info"],
                             args_map["gt_boxes"], args_map["label"],
                             args_map["bbox_target"], args_map["bbox_weight"]);
          CheckMXnetError("copy batch");

          // monitor.tic();
          executor->Forward(true);
          mxnet::cpp::NDArray::WaitAll();
          CheckMXnetError("forward");

          // evaluate training metrics - every 100 batches
          if (batch_num % 100 == 0) {
            rpn_acc_metric.Reset();
            rpn_log_loss_metric.Reset();
            rpn_l1_loss_metric.Reset();

            rcnn_acc_metric.Reset();
            rcnn_log_loss_metric.Reset();
            rcnn_l1_loss_metric.Reset();

            CheckMXnetError("forward metrics");
            rcnn_acc_metric.Update(executor->outputs[4], executor->outputs[2]);
            CheckMXnetError("metric 1");
            rcnn_log_loss_metric.Update(executor->outputs[4],
                                        executor->outputs[2]);
            CheckMXnetError("metric 2");
            rcnn_l1_loss_metric.Update(executor->outputs[4],
                                       executor->outputs[3]);
            CheckMXnetError("metric 3");

            rpn_acc_metric.Update(args_map["label"], executor->outputs[0]);
            CheckMXnetError("metric 4");
            rpn_log_loss_metric.Update(args_map["label"], executor->outputs[0]);
            CheckMXnetError("metric 5");
            rpn_l1_loss_metric.Update(args_map["bbox_weight"],
                                      executor->outputs[1]);
            CheckMXnetError("metric 6");

            reporter.SetLineValue(2, rpn_acc_metric.Get());
            reporter.SetLineValue(3, rpn_log_loss_metric.Get());
            reporter.SetLineValue(4, rpn_l1_loss_metric.Get());
            reporter.SetLineValue(5, rcnn_acc_metric.Get());
            reporter.SetLineValue(6, rcnn_log_loss_metric.Get());
            reporter.SetLineValue(7, rcnn_l1_loss_metric.Get());
          }

          executor->Backward();
          mxnet::cpp::NDArray::WaitAll();
          CheckMXnetError("backward");
          // monitor.toc_print();

          for (size_t i = 0; i < args.size(); ++i) {
            if (not_update_args.find(args[i]) == not_update_args.end()) {
              opt->Update(static_cast<int>(i), executor->arg_arrays[i],
                          executor->grad_arrays[i]);
            }
          }
          mxnet::cpp::NDArray::WaitAll();
          CheckMXnetError("updates");

          ++batch_num;
        }
        SaveNetParams(check_point_file, executor);
        std::cout << "Parameters saved to " << check_point_file << std::endl;
      }
      reporter.Stop();

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
