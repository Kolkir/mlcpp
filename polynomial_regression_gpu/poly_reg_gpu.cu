/* This sample is based on the Chapter 1 from book
 * "Building Machine Learning Systems with Python" by Willi Richert
 */

// third party includes
#include <csv.h>
#include <mshadow/tensor.h>
#include <plot.h>

// stl includes
#include <experimental/filesystem>
#include <iostream>
#include <random>
#include <string>

// application includes
#include "../ioutils.h"
#include "../utils.h"

// Namespace and type aliases
namespace fs = std::experimental::filesystem;
namespace ms = mshadow;
typedef float DType;

template <typename Device>
struct ScopedTensorEngine {
  ScopedTensorEngine() { ms::InitTensorEngine<Device>(); }
  ~ScopedTensorEngine() { ms::ShutdownTensorEngine<Device>(); }
  ScopedTensorEngine(const ScopedTensorEngine&) = delete;
  ScopedTensorEngine& operator=(const ScopedTensorEngine&) = delete;
};

struct Pow {
  MSHADOW_XINLINE static float Map(float x, float y) { return pow(x, y); }
};

int main() {
  // Download the data
  const std::string data_path{"web_traffic.tsv"};
  if (!fs::exists(data_path)) {
    const std::string data_url{
        R"(https://raw.githubusercontent.com/luispedro/BuildingMachineLearningSystemsWithPython/master/ch01/data/web_traffic.tsv)"};
    if (!utils::DownloadFile(data_url, data_path)) {
      std::cerr << "Unable to download the file " << data_url << std::endl;
      return 1;
    }
  }

  // Read the data
  io::CSVReader<2, io::trim_chars<' '>, io::no_quote_escape<'\t'>> data_tsv(
      data_path);

  std::vector<DType> raw_data_x;
  std::vector<DType> raw_data_y;

  bool done = false;
  do {
    try {
      DType x = 0, y = 0;
      done = !data_tsv.read_row(x, y);
      if (!done) {
        raw_data_x.push_back(x);
        raw_data_y.push_back(y);
      }
    } catch (const io::error::no_digit& err) {
      // ignore bad formated samples
      std::cout << err.what() << std::endl;
    }
  } while (!done);

  // shuffle data
  size_t seed = 25345;
  std::shuffle(raw_data_x.begin(), raw_data_x.end(),
               std::default_random_engine(seed));
  std::shuffle(raw_data_y.begin(), raw_data_y.end(),
               std::default_random_engine(seed));

  ScopedTensorEngine<ms::cpu> tensorEngineCpu;
  ScopedTensorEngine<ms::gpu> tensorEngineGpu;
  ms::Stream<ms::gpu>* computeStream = ms::NewStream<ms::gpu>(true, false, -1);

  // map data to the tensors
  auto rows = raw_data_x.size();
  ms::Tensor<ms::cpu, 1, DType> host_y(raw_data_y.data(), ms::Shape1(rows));  
  ms::TensorContainer<ms::gpu, 1, DType> gpu_y(host_y.shape_);
  gpu_y.set_stream(computeStream);
  ms::Copy(gpu_y, host_y, computeStream);

  ms::Tensor<ms::cpu, 2, DType> host_x(raw_data_x.data(), ms::Shape2(rows, 1));
  ms::TensorContainer<ms::gpu, 2, DType> gpu_basis_x(host_x.shape_);
  gpu_basis_x.set_stream(computeStream);
  ms::Copy(gpu_basis_x, host_x, computeStream);

  // min-max scaling
  auto minmax_y = std::minmax_element(raw_data_y.begin(), raw_data_y.end());
  gpu_y = (gpu_y - *minmax_y.first) / (*minmax_y.second - *minmax_y.first);

  auto minmax_x = std::minmax_element(raw_data_x.begin(), raw_data_x.end());
  gpu_basis_x =
      (gpu_basis_x - *minmax_x.first) / (*minmax_x.second - *minmax_x.first);

  // generate polynom
  const size_t p_degree = 12;
  ms::TensorContainer<ms::gpu, 2, DType> gpu_x(ms::Shape2(rows, p_degree));
  gpu_x.set_stream(computeStream);
  for (size_t c = 0; c < p_degree; ++c) {
    auto col =
        ms::expr::slice(gpu_x, ms::Shape2(0, c), ms::Shape2(rows, c + 1));
    col = ms::expr::F<Pow>(gpu_basis_x, c);
  }
  std::cout << gpu_x.shape_[0] << " " <<  gpu_x.shape_[1] << std::endl;

  // learn polynomial regression with Batch Gradient Descent
  size_t n_epochs = 1000;
  size_t batch_size = 15;
  size_t n_batches = raw_data_y.size() / batch_size;
  DType lr = 0.1;

  // it is important to allocate all tensors before assiging
  ms::TensorContainer<ms::gpu, 1, DType> gpu_weights(ms::Shape1(p_degree));
  gpu_weights.set_stream(computeStream);
  gpu_weights = 0.0f;

  ms::TensorContainer<ms::gpu, 1, DType> gpu_grad(ms::Shape1(p_degree));
  gpu_grad.set_stream(computeStream);

  ms::TensorContainer<ms::gpu, 1, DType> yhat(ms::Shape1(batch_size));
  yhat.set_stream(computeStream);

  ms::TensorContainer<ms::gpu, 1, DType> error(ms::Shape1(batch_size));
  error.set_stream(computeStream);

  ms::TensorContainer<ms::gpu, 1, DType> error_total(ms::Shape1(rows));
  error_total.set_stream(computeStream);

  ms::TensorContainer<ms::cpu, 1, DType> error_total_cpu(ms::Shape1(rows));

  // gradient descent
  for (size_t epoch = 0; epoch < n_epochs; ++epoch) {
    for (size_t bi = 0; bi < n_batches; ++bi) {
      auto bs = bi * batch_size;
      auto be = bs + batch_size;
      auto batch_x = gpu_x.Slice(bs, be);
      auto batch_y = gpu_y.Slice(bs, be);

      yhat = ms::expr::sumall_except_dim<0>(ms::expr::broadcast<1>(gpu_weights, batch_x.shape_) * batch_x);
      error = yhat - batch_y;
      gpu_grad = ms::expr::dot(error, batch_x);
      gpu_grad /= batch_size;
      gpu_weights = gpu_weights - (lr * gpu_grad);
    }
    //compute cost
    error_total = ms::expr::sumall_except_dim<0>(ms::expr::broadcast<1>(gpu_weights, gpu_x.shape_) * gpu_x);
    error_total = error_total - gpu_y;
    error_total = error_total * error_total;
    ms::Copy(error_total_cpu, error_total, computeStream);
    long double cost = 0;
    for (size_t r = 0; r < rows; ++r)
      cost += error_total_cpu[r];
    cost /= rows;
    std::cout << "Epoch " << epoch << " cost = " << cost << std::endl;
  }

  // make predictions
  gpu_y = ms::expr::sumall_except_dim<0>(ms::expr::broadcast<1>(gpu_weights, gpu_x.shape_) * gpu_x);

  // restore scaling
  gpu_y = gpu_y * (*minmax_y.second - *minmax_y.first) + *minmax_y.first;

  // get results from gpu
  ms::TensorContainer<ms::cpu, 1, DType> pred_y(ms::Shape1(rows));
  ms::Copy(pred_y, gpu_y, computeStream);

  // free resources
  ms::DeleteStream(computeStream);

  // plot the data we read and approximate
  plotcpp::Plot plt(true);
  plt.SetTerminal("qt");
  plt.SetTitle("Web traffic over the last month");
  plt.SetXLabel("Time");
  plt.SetYLabel("Hits/hour");
  plt.SetAutoscale();
  plt.GnuplotCommand("set grid");

  auto time_range = *minmax_x.second - *minmax_x.first;
  auto tic_size = 7.f * 24.f;
  auto time_tics = time_range / tic_size;
  plt.SetXRange(-tic_size / 2, *minmax_x.second  + tic_size / 2);

  plotcpp::Plot::Tics xtics;
  for (size_t t = 0; t < time_tics; ++t) {
    xtics.push_back({"week " + std::to_string(t), t * tic_size});
  }
  plt.SetXTics(xtics);

  plt.Draw2D(plotcpp::Points(raw_data_x.begin(), raw_data_x.end(), raw_data_y.begin(),
                             "points", "lc rgb 'black' pt 1"),
             plotcpp::Points(raw_data_x.begin(), raw_data_x.end(), pred_y.dptr_,
                            "poly line approx", "lc rgb 'green' lw 2"));
  plt.Flush();
  return 0;
}
