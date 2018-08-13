#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

/*The global context, change them if necessary*/
static mxnet::cpp::Context global_ctx(mxnet::cpp::kGPU, 0);
// static Context global_ctx(mxnet::cpp::kCPU,0);

int _main_(int argc, char** argv) {
  using namespace mxnet::cpp;
  if (argc > 1) {
    //---------- Load symbols
    std::string symbol_file = "../model/model_symbol.json";
    Symbol net = Symbol::Load(symbol_file).GetInternals();
    for (const auto& layer_name : net.ListOutputs()) {
      LG << layer_name;
    }

    net = Symbol::Load(symbol_file).GetInternals()["softmax_output"];
    //---------- Load parameters
    std::string param_file = "../model/model_param.params";
    std::map<std::string, NDArray> paramters;
    NDArray::Load(param_file, nullptr, &paramters);
    std::map<std::string, NDArray> args_map;
    std::map<std::string, NDArray> aux_map;
    for (const auto& k : paramters) {
      if (k.first.substr(0, 4) == "aux:") {
        auto name = k.first.substr(4, k.first.size() - 4);
        aux_map[name] = k.second.Copy(global_ctx);
      }
      if (k.first.substr(0, 4) == "arg:") {
        auto name = k.first.substr(4, k.first.size() - 4);
        args_map[name] = k.second.Copy(global_ctx);
      }
    }
    /*WaitAll is need when we copy data between GPU and the main memory*/
    NDArray::WaitAll();

    //----------- Load mean
    NDArray mean_img(Shape(1, 3, 224, 224), global_ctx, false);
    mean_img.SyncCopyFromCPU(
        NDArray::LoadToMap("../model/mean_224.nd")["mean_img"].GetData(),
        1 * 3 * 224 * 224);
    NDArray::WaitAll();

    //----------- Load data
    NDArray data(Shape(1, 3, 224, 224), global_ctx, false);
    std::vector<float> array(data.Size());
    cv::Mat mat = cv::imread(argv[1]);
    /*resize pictures to (224, 224) according to the pretrained model*/
    cv::resize(mat, mat, cv::Size(224, 224));
    size_t k = 0;
    for (size_t c = 0; c < 3; ++c) {
      for (size_t i = 0; i < 224; ++i) {
        for (size_t j = 0; j < 224; ++j) {
          auto t = (i * 224 + j) * 3 + c;
          array[k] = static_cast<float>(mat.data[t]);
          ++k;
        }
      }
    }
    data.SyncCopyFromCPU(array.data(), array.size());
    NDArray::WaitAll();

    // Disable for resnet
    // data -= mean_img;
    args_map["data"] = data;

    //----------- Predict
    Executor* executor =
        net.SimpleBind(global_ctx, args_map, std::map<std::string, NDArray>(),
                       std::map<std::string, OpReqType>(), aux_map);
    executor->Forward(false);
    auto prediction = executor->outputs[0].Copy(Context(kCPU, 0));
    NDArray::WaitAll();
    delete executor;

    //----------- Show results
    float max_pred = 0;
    size_t max_index = 0;

    auto shape = prediction.GetShape();

    for (size_t i = 0; i < prediction.Size(); ++i) {
      if (prediction.At(0, i) > max_pred) {
        max_pred = prediction.At(0, i);
        max_index = i;
      }
    }
    std::vector<std::string> labels;
    {
      std::ifstream labels_file("../model/synset.txt");
      if (labels_file) {
        std::string label;
        while (std::getline(labels_file, label)) {
          labels.push_back(label);
        }
      }
    }
    std::cout << "Probalility " << max_pred << " Class " << labels[max_index]
              << std::endl;
  }
  MXNotifyShutdown();
  return 0;
}
