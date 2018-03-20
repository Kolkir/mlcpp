/* This sample is based on the Chapter 1 from book
 * "Building Machine Learning Systems with Python" by Willi Richert
 */

#include <csv.h>
#include <plot.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xio.hpp>

// regular includes
#include <experimental/filesystem>
#include <iostream>
#include <string>
#include "../utils.h"

// Namespace and type aliases
using namespace std::literals::string_literals;
namespace fs = std::experimental::filesystem;

template <typename S, typename T>
void PrintShape(const S& name, const T& shape) {
  std::cout << name << " {";
  for (size_t i = 0; i < shape.size(); ++i) {
    std::cout << shape[i];
    if (i < shape.size() - 1)
      std::cout << ", ";
  }
  std::cout << "}\n";
  std::cout.flush();
}

template <typename Vb, typename Vx>
auto predict(const Vb& b, const Vx& x) {
  return xt::sum(x * b, {1});
}

template <typename V>
auto minmax_scale(const V& v) {
  if (v.shape().size() == 1) {
    auto minmax = xt::minmax(v)();
    xt::xarray<typename V::value_type> vs =
        (v - minmax[0]) / (minmax[1] - minmax[0]);
    return vs;
  } else if (v.shape().size() == 2) {
    auto w = v.shape()[1];
    xt::xarray<typename V::value_type> vs =
        xt::zeros<typename V::value_type>(v.shape());
    for (decltype(w) j = 0; j < w; ++j) {
      auto vc = xt::view(v, xt::all(), j);
      auto vsc = xt::view(vs, xt::all(), j);
      auto minmax = xt::minmax(vc)();
      vsc = (vc - minmax[0]) / (minmax[1] - minmax[0]);
    }
    return vs;
  } else {
    throw std::logic_error("Minmax scale unsupported dimensions");
  }
}

template <typename Vx, typename Vy>
auto gd(const Vx& x, const Vy& y) {
  size_t n_epochs = 1000;
  float lr = 0.001;
  auto n = x.shape()[0];
  auto w = x.shape()[1];

  xt::xarray<typename Vx::value_type> b =
      xt::zeros<typename Vx::value_type>({w});
  xt::xarray<typename Vx::value_type> grad =
      xt::zeros<typename Vx::value_type>({w});

  for (size_t i = 0; i < n_epochs; ++i) {
    auto yhat = predict(b, x);
    auto error = yhat - y;
    for (size_t j = 0; j < w; ++j) {
      auto xc = xt::view(x, xt::all(), j);
      grad[j] = (xt::sum(xc * error) / n)();
    }
    b = b - lr * grad;
    auto cost = (xt::sum(xt::pow(error, 2)) / n)();
    std::cout << "Iteration : " << i << " Cost = " << cost << " b0 = " << b[0]
              << " b1 = " << b[1] << std::endl;
  }
  return b;
}

template <typename Vx, typename Vy>
auto straight_line_model(const Vx& data_x, const Vy& data_y) {
  // minmax scaling
  auto y = minmax_scale(data_y);

  xt::xarray<float> x = xt::zeros<float>(data_x.shape());
  x = data_x;
  auto xv = xt::view(x, xt::all(), xt::range(1, xt::placeholders::_));
  xv = minmax_scale(xv);

  // learn line parameters with Gradient Descent
  auto b = gd(x, y);

  // model with line
  xt::xarray<float> line_values = predict(b, x);

  // restore scaling for predicted line values
  auto y_minmax = xt::minmax(data_y)();
  line_values = line_values * (y_minmax[1] - y_minmax[0]) + y_minmax[0];
  return line_values;
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
  const size_t cols = 2;
  io::CSVReader<cols, io::trim_chars<' '>, io::no_quote_escape<'\t'>> data_tsv(
      data_path);

  std::vector<float> raw_data_x;
  std::vector<float> raw_data_y;

  bool done = false;
  do {
    try {
      float x = 0, y = 0;
      done = !data_tsv.read_row(x, y);
      raw_data_x.push_back(1);  // additional x for simpler vectorization
      raw_data_x.push_back(x);
      raw_data_y.push_back(y);
    } catch (const io::error::no_digit& err) {
      // ignore bad formated samples
      std::cout << err.what() << std::endl;
    }
  } while (!done);

  // map data to the tensor
  size_t rows = raw_data_x.size() / 2;
  auto shape_x = std::vector<size_t>{rows, size_t{2}};
  auto data_x = xt::adapt(raw_data_x, shape_x);
  PrintShape("X"s, data_x.shape());

  auto shape_y = std::vector<size_t>{rows};
  auto data_y = xt::adapt(raw_data_y, shape_y);
  PrintShape("Y"s, data_y.shape());

  xt::xarray<float> line_values = straight_line_model(data_x, data_y);

  // plot the data we read
  plotcpp::Plot plt(true);
  plt.SetTerminal("qt");
  plt.SetTitle("Web traffic over the last month");
  plt.SetXLabel("Time");
  plt.SetYLabel("Hits/hour");
  plt.SetAutoscale();
  plt.GnuplotCommand("set grid");

  auto minmax = std::minmax_element(data_x.begin(), data_x.end());
  auto diff = *minmax.second - *minmax.first;
  auto tic_size = 7 * 24;
  auto time_tics = diff / tic_size;

  plt.GnuplotCommand("set xrange [" + std::to_string(-tic_size / 2) + ":" +
                     std::to_string(*minmax.second + tic_size / 2) + "]");
  std::stringstream xtics_labels;
  xtics_labels << "set xtics (";
  for (size_t t = 0; t < time_tics; ++t) {
    xtics_labels << R"("week )" << t << R"(" )" << t * tic_size << ",";
  }
  xtics_labels << ")";
  plt.GnuplotCommand(xtics_labels.str());

  auto x_coord = xt::view(data_x, xt::all(), 1);
  plt.Draw2D(
      plotcpp::Points(x_coord.begin(), x_coord.end(), data_y.begin(), "points"),
      plotcpp::Lines(x_coord.begin(), x_coord.end(), line_values.begin(),
                     "line approx"));
  plt.Flush();

  return 0;
}
