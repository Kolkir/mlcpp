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
#include <tuple>

// application includes
#include "../ioutils.h"
#include "../utils.h"

// Namespace and type aliases
namespace fs = std::experimental::filesystem;
namespace ms = mshadow;
using GpuStream = ms::Stream<ms::gpu>;
using GpuStreamPtr = std::unique_ptr<GpuStream, void (*)(GpuStream*)>;
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

struct Sqrt {
  MSHADOW_XINLINE static float Map(float x) { return sqrt(x); }
};

template <typename Device>
class Standardizer {
 public:
  using Tensor = ms::TensorContainer<Device, 2, DType>;
  using Stream = ms::Stream<Device>;

  Standardizer(Stream* computeStream, size_t rows)
      : min(ms::Shape1(1)),
        max(ms::Shape1(1)),
        mean(ms::Shape1(1)),
        temp(ms::Shape2(rows, 1)),
        sd(ms::Shape1(1)),
        rows(rows) {
    min.set_stream(computeStream);
    max.set_stream(computeStream);
    mean.set_stream(computeStream);
    sd.set_stream(computeStream);
    temp.set_stream(computeStream);
  }
  ~Standardizer() {}
  Standardizer(const Standardizer&) = delete;
  Standardizer& operator=(const Standardizer&) = delete;

  void standardize(Tensor& vec, Stream* computeStream) {
    mean = ms::expr::sumall_except_dim<1>(vec);
    mean /= static_cast<DType>(rows);
    temp = ms::expr::F<Pow>(vec - ms::expr::broadcast<1>(mean, temp.shape_), 2);
    sd = ms::expr::sumall_except_dim<1>(temp);
    sd = ms::expr::F<Sqrt>(sd) / static_cast<DType>(rows - 1);
    temp = (vec - ms::expr::broadcast<1>(mean, temp.shape_)) /
           ms::expr::broadcast<1>(sd, temp.shape_);

    // scale to [-1, 1] range
    min = ms::expr::ReduceTo1DExp<Tensor, DType, ms::red::minimum,
                                  ms::expr::ExpInfo<Tensor>::kDim - 1>(
        temp, DType(1));
    max = ms::expr::ReduceTo1DExp<Tensor, DType, ms::red::maximum,
                                  ms::expr::ExpInfo<Tensor>::kDim - 1>(
        temp, DType(1));

    temp = (temp - ms::expr::broadcast<1>(min, temp.shape_)) /
           ms::expr::broadcast<1>(max - min, temp.shape_);

    temp = (temp * 2.f) - 1.f;

    ms::Copy(vec, temp, computeStream);
  }

  auto get_moments(Stream* computeStream) {
    ms::TensorContainer<ms::cpu, 1, DType> value(ms::Shape1(1));
    ms::Copy(value, min, computeStream);
    DType v_min = value[0];
    ms::Copy(value, max, computeStream);
    DType v_max = value[0];
    ms::Copy(value, mean, computeStream);
    DType v_mean = value[0];
    ms::Copy(value, sd, computeStream);
    DType v_sd = value[0];
    return std::vector<DType>{v_min, v_max, v_mean, v_sd};
  }

 private:
  ms::TensorContainer<Device, 1, DType> min;
  ms::TensorContainer<Device, 1, DType> max;
  ms::TensorContainer<Device, 1, DType> mean;
  ms::TensorContainer<Device, 1, DType> sd;
  ms::TensorContainer<Device, 2, DType> temp;
  size_t rows;
};

template <typename Device>
void generate_polynomial(std::vector<DType>& raw_data_x,
                         size_t p_degree,
                         ms::TensorContainer<Device, 2, DType>& out_x,
                         ms::Stream<Device>* computeStream) {
  auto rows = raw_data_x.size();
  ms::Tensor<ms::cpu, 2, DType> host_x(raw_data_x.data(), ms::Shape2(rows, 1));
  ms::TensorContainer<Device, 2, DType> gpu_basis_x(host_x.shape_);
  gpu_basis_x.set_stream(computeStream);
  ms::Copy(gpu_basis_x, host_x, computeStream);

  // standardize / normalize
  Standardizer<Device> standardizer(computeStream, rows);
  standardizer.standardize(gpu_basis_x, computeStream);

  ms::TensorContainer<ms::gpu, 2, DType> col_temp(ms::Shape2(rows, 1));
  col_temp.set_stream(computeStream);
  out_x.set_stream(computeStream);
  for (size_t c = 0; c < p_degree; ++c) {
    auto col =
        ms::expr::slice(out_x, ms::Shape2(0, c), ms::Shape2(rows, c + 1));
    col_temp = ms::expr::F<Pow>(gpu_basis_x, static_cast<DType>(c));
    if (c > 1)
      standardizer.standardize(col_temp, computeStream);
    col = col_temp;
  }
}

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
  GpuStreamPtr computeStream(ms::NewStream<ms::gpu>(true, false, -1),
                             [](GpuStream* s) { ms::DeleteStream(s); });

  // map data to the tensors
  auto rows = raw_data_x.size();
  ms::Tensor<ms::cpu, 2, DType> host_y(raw_data_y.data(), ms::Shape2(rows, 1));
  ms::TensorContainer<ms::gpu, 2, DType> gpu_y(host_y.shape_);
  gpu_y.set_stream(computeStream.get());
  ms::Copy(gpu_y, host_y, computeStream.get());

  // standardize / normalize
  Standardizer<ms::gpu> standardizer(computeStream.get(), rows);
  standardizer.standardize(gpu_y, computeStream.get());
  auto y_moments = standardizer.get_moments(computeStream.get());

  // generate polynom
  const size_t p_degree = 64;
  ms::TensorContainer<ms::gpu, 2, DType> gpu_x(ms::Shape2(rows, p_degree));
  generate_polynomial(raw_data_x, p_degree, gpu_x, computeStream.get());

  // learn polynomial regression with Batch Gradient Descent
  size_t n_epochs = 5000;
  size_t batch_size = 8;
  size_t n_batches = raw_data_y.size() / batch_size;
  // DType lr = 0.0055; // BGD
  DType lr = 0.99;
  DType e = std::numeric_limits<DType>::epsilon();

  // it is important to allocate all tensors before assiging
  ms::TensorContainer<ms::gpu, 2, DType> gpu_weights(ms::Shape2(p_degree, 1));
  gpu_weights.set_stream(computeStream.get());
  gpu_weights = 0.0f;
  ms::TensorContainer<ms::cpu, 2, DType> cpu_weights(ms::Shape2(p_degree, 1));

  ms::TensorContainer<ms::gpu, 2, DType> gpu_grad(ms::Shape2(p_degree, 1));
  gpu_grad.set_stream(computeStream.get());

  ms::TensorContainer<ms::gpu, 2, DType> yhat(ms::Shape2(batch_size, 1));
  yhat.set_stream(computeStream.get());

  ms::TensorContainer<ms::gpu, 2, DType> error(ms::Shape2(batch_size, 1));
  error.set_stream(computeStream.get());

  ms::TensorContainer<ms::gpu, 2, DType> error_total(ms::Shape2(rows, 1));
  error_total.set_stream(computeStream.get());

  ms::TensorContainer<ms::cpu, 2, DType> error_total_cpu(ms::Shape2(rows, 1));

  ms::TensorContainer<ms::gpu, 2, DType> gpu_eg_sum(ms::Shape2(p_degree, 1));
  gpu_eg_sum.set_stream(computeStream.get());
  gpu_eg_sum = 0.f;
  ms::TensorContainer<ms::gpu, 2, DType> gpu_weights_delta(
      ms::Shape2(p_degree, 1));
  gpu_weights_delta.set_stream(computeStream.get());
  ms::TensorContainer<ms::gpu, 2, DType> gpu_ex_sum(ms::Shape2(p_degree, 1));
  gpu_ex_sum.set_stream(computeStream.get());
  gpu_ex_sum = 0.f;

  // gradient descent
  for (size_t epoch = 0; epoch < n_epochs; ++epoch) {
    for (size_t bi = 0; bi < n_batches; ++bi) {
      auto bs = bi * batch_size;
      auto be = bs + batch_size;
      auto batch_x = gpu_x.Slice(bs, be);
      auto batch_y = gpu_y.Slice(bs, be);

      yhat = ms::expr::dot(batch_x, gpu_weights);
      error = yhat - batch_y;
      gpu_grad = ms::expr::dot(batch_x.T(), error);
      gpu_grad /= batch_size;

      // AdaDelta
      gpu_eg_sum = lr * gpu_eg_sum + (1.f - lr) * ms::expr::F<Pow>(gpu_grad, 2);
      gpu_weights_delta = -1.f *
                          (ms::expr::F<Sqrt>(gpu_ex_sum + e) /
                           ms::expr::F<Sqrt>(gpu_eg_sum + e)) *
                          gpu_grad;
      gpu_ex_sum =
          lr * gpu_ex_sum + (1.f - lr) * ms::expr::F<Pow>(gpu_weights_delta, 2);
      gpu_weights = gpu_weights + gpu_weights_delta;

      // BGD
      // gpu_weights = gpu_weights - (lr * gpu_grad);

      // Print weights
      //      ms::Copy(cpu_weights, gpu_weights, computeStream.get());
      //      std::cout << "weights : ";
      //      for (size_t r = 0; r < p_degree; ++r)
      //        std::cout << cpu_weights[0][r] << "; ";
      //      std::cout << std::endl;
    }
    // compute cost
    error_total = ms::expr::dot(gpu_x, gpu_weights);
    error_total = ms::expr::F<Pow>(error_total - gpu_y, 2);
    ms::Copy(error_total_cpu, error_total, computeStream.get());
    long double cost = 0;
    for (size_t r = 0; r < rows; ++r)
      cost += error_total_cpu[r][0];
    cost /= rows;
    std::cout << "Epoch " << epoch << " cost = " << cost << std::endl;
  }

  // generate new data
  size_t n = 2000;
  auto minmax_x = std::minmax_element(raw_data_x.begin(), raw_data_x.end());
  auto time_range = *minmax_x.second - *minmax_x.first;
  auto inc_step = time_range / n;
  auto x_val = inc_step;
  std::vector<DType> new_data_x(n);
  for (auto& x : new_data_x) {
    x = x_val;
    x_val += inc_step;
  }
  ms::TensorContainer<ms::gpu, 2, DType> new_gpu_x(ms::Shape2(n, p_degree));
  generate_polynomial(new_data_x, p_degree, new_gpu_x, computeStream.get());

  // make predictions
  ms::TensorContainer<ms::gpu, 2, DType> new_gpu_y(ms::Shape2(n, 1));
  new_gpu_y.set_stream(computeStream.get());
  new_gpu_y = ms::expr::dot(new_gpu_x, gpu_weights);

  // restore scaling
  new_gpu_y = (new_gpu_y + 1.f) / DType(2);
  new_gpu_y = new_gpu_y * (y_moments[1] - y_moments[0]) + y_moments[0];
  new_gpu_y = (new_gpu_y * y_moments[3]) + y_moments[2];

  // get results from gpu
  std::vector<DType> raw_pred_y(n);
  ms::Tensor<ms::cpu, 2, DType> pred_y(raw_pred_y.data(), ms::Shape2(n, 1));
  ms::Copy(pred_y, new_gpu_y, computeStream.get());

  // plot the data we read and approximate
  plotcpp::Plot plt(true);
  plt.SetTerminal("qt");
  plt.SetTitle("Web traffic over the last month");
  plt.SetXLabel("Time");
  plt.SetYLabel("Hits/hour");
  plt.SetAutoscale();
  plt.GnuplotCommand("set grid");

  auto tic_size = 7.f * 24.f;
  auto time_tics = time_range / tic_size;
  plt.SetXRange(-tic_size / 2, *minmax_x.second + tic_size / 2);

  plotcpp::Plot::Tics xtics;
  for (size_t t = 0; t < time_tics; ++t) {
    xtics.push_back({"week " + std::to_string(t), t * tic_size});
  }
  plt.SetXTics(xtics);

  plt.Draw2D(
      plotcpp::Points(raw_data_x.begin(), raw_data_x.end(), raw_data_y.begin(),
                      "points", "lc rgb 'black' pt 1"),
      plotcpp::Lines(new_data_x.begin(), new_data_x.end(), raw_pred_y.begin(),
                     "poly line approx", "lc rgb 'green' lw 2"));
  plt.Flush();

  return 0;
}
