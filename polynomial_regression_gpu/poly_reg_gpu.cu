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
#include "common.h"
#include "optimizer.h"
#include "polynomial.h"
#include "standardizer.h"

#define USE_GPU

// Namespace and type aliases
namespace fs = std::experimental::filesystem;
#ifdef USE_GPU  // use macros because lack of "if constexpr"
using xpu = mshadow::gpu;
#else
using xpu = mshadow::cpu;
#endif
using DType = float;
using GpuStream = mshadow::Stream<xpu>;
using GpuStreamPtr = std::unique_ptr<GpuStream, void (*)(GpuStream*)>;

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

  // define comupte engines
  ScopedTensorEngine<mshadow::cpu> tensorEngineCpu;
#ifdef USE_GPU  // use macros because lack of "if constexpr"
  ScopedTensorEngine<mshadow::gpu> tensorEngineGpu;
#endif
  GpuStreamPtr computeStream(mshadow::NewStream<xpu>(true, false, -1),
                             [](GpuStream* s) { mshadow::DeleteStream(s); });

  // map data to the tensors
  mshadow::TensorContainer<xpu, 2, DType> x;
  x.set_stream(computeStream.get());
  load_data<xpu>(raw_data_x, x);

  mshadow::TensorContainer<xpu, 2, DType> y;
  y.set_stream(computeStream.get());
  load_data<xpu>(raw_data_y, y);

  // standardize data
  auto rows = raw_data_x.size();
  Standardizer<xpu, DType> standardizer;
  standardizer.transform(x);
  standardizer.transform(y);
  auto y_moments = standardizer.get_moments();

  // we need to scale the data to an appropriate range before raise to power
  // elements to prevent float overflow in the optimizer.
  DType scale = 0.6;
  x *= scale;
  y *= scale;

  // generate polynom
  size_t p_degree = 64;
  mshadow::TensorContainer<xpu, 2, DType> poly_x(
      mshadow::Shape2(rows, p_degree));
  poly_x.set_stream(computeStream.get());
  generate_polynomial(x, poly_x, p_degree);

  // learn polynomial regression with Batch Gradient Descent
  Optimizer<xpu, DType> optimizer;
  optimizer.fit(poly_x, y);

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
  mshadow::TensorContainer<xpu, 2, DType> new_x(mshadow::Shape2(n, 1));
  new_x.set_stream(computeStream.get());
  load_data<xpu>(new_data_x, new_x);
  standardizer.transform(new_x);
  new_x *= scale;

  mshadow::TensorContainer<xpu, 2, DType> new_poly_x(
      mshadow::Shape2(n, p_degree));
  new_poly_x.set_stream(computeStream.get());
  generate_polynomial(new_x, new_poly_x, p_degree);

  // make predictions
  mshadow::TensorContainer<xpu, 2, DType> new_y(mshadow::Shape2(n, 1));
  new_y.set_stream(computeStream.get());
  optimizer.predict(new_poly_x, new_y);

  // restore scaling
  new_y /= scale;
  new_y = (new_y * y_moments[1]) + y_moments[0];

  // get results from gpu
  std::vector<DType> raw_pred_y(n);
  mshadow::Tensor<mshadow::cpu, 2, DType> pred_y(raw_pred_y.data(),
                                                 mshadow::Shape2(n, 1));
  mshadow::Copy(pred_y, new_y, computeStream.get());

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
