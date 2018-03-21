/* This sample is based on the Chapter 1 from book
 * "Building Machine Learning Systems with Python" by Willi Richert
 */

// third party includes
#include <csv.h>
#include <plot.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xio.hpp>

// stl includes
#include <experimental/filesystem>
#include <iostream>
#include <string>

// application includes
#include "../ioutils.h"
#include "../utils.h"

// Namespace and type aliases
namespace fs = std::experimental::filesystem;
typedef float DType;

template <typename V>
auto generate_polynom(const V& v, size_t degree) {
  assert(v.shape().size() == 1);
  auto rows = v.shape()[0];
  auto poly_shape = std::vector<size_t>{rows, degree};
  xt::xarray<DType> poly_x = xt::zeros<DType>(poly_shape);
  // fill additional column for simpler vectorization
  {
    auto xv = xt::view(poly_x, xt::all(), 0);
    xv = xt::ones<DType>({rows});
  }
  // copy initial data
  {
    auto xv = xt::view(poly_x, xt::all(), 1);
    xv = v;
  }
  // generate additional terms
  auto x_col = xt::view(poly_x, xt::all(), 1);
  for (size_t i = 2; i < degree; ++i) {
    auto xv = xt::view(poly_x, xt::all(), i);
    xv = xt::pow(x_col, i);
  }
  return poly_x;
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

template <typename Vb, typename Vx>
auto predict(const Vb& b, const Vx& x) {
  return xt::sum(b * x, {1});
}

template <typename Vx, typename Vy>
auto gd(const Vx& x, const Vy& y) {
  size_t n_epochs = 15000;
  DType lr = 0.001;
  auto n = x.shape()[0];
  auto w = x.shape()[1];

  xt::xarray<typename Vx::value_type> b =
      xt::zeros<typename Vx::value_type>({w});
  xt::xarray<typename Vx::value_type> grad =
      xt::zeros<typename Vx::value_type>({w});

  DType cost = std::numeric_limits<DType>::max();
  for (size_t i = 0; i < n_epochs; ++i) {
    auto yhat = predict(b, x);
    auto error = yhat - y;
    for (size_t j = 0; j < w; ++j) {
      auto xc = xt::view(x, xt::all(), j);
      grad[j] = (xt::sum(xc * error) / n)();
    }
    b = b - lr * grad;
    cost = (xt::sum(xt::pow(error, 2)) / n)();
    // std::cout << "Iteration : " << i << " Cost = " << cost  << std::endl;
  }
  std::cout << "Final cost = " << cost << std::endl;
  return b;
}

template <typename V>
auto adapt_x(const V& data_x, size_t p_degree) {
  xt::xarray<DType> poly_x = generate_polynom(data_x, p_degree);
  xt::xarray<DType> x = xt::zeros<DType>(poly_x.shape());
  x = poly_x;
  auto xv = xt::view(x, xt::all(), xt::range(1, xt::placeholders::_));
  xv = minmax_scale(xv);
  return x;
}

template <typename Vx, typename Vy>
auto make_regression_model(const Vx& data_x,
                           const Vy& data_y,
                           size_t p_degree) {
  // minmax scaling
  auto y = xt::eval(minmax_scale(data_y));

  // minmax scaling & polynomization
  auto x = xt::eval(adapt_x(data_x, p_degree));

  // learn parameters with Gradient Descent
  auto b = gd(x, y);

  // create model
  auto y_minmax = xt::minmax(data_y)();
  auto model = [b, y_minmax, p_degree](const auto& data_x) {
    auto x = xt::eval(adapt_x(data_x, p_degree));
    xt::xarray<DType> yhat = xt::eval(predict(b, x));

    // restore scaling for predicted line values

    yhat = yhat * (y_minmax[1] - y_minmax[0]) + y_minmax[0];
    return yhat;
  };
  return model;
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

  // map data to the tensor
  size_t rows = raw_data_x.size();
  auto shape_x = std::vector<size_t>{rows};
  auto data_x = xt::adapt(raw_data_x, shape_x);
  std::cout << "X shape " << data_x.shape() << std::endl;

  auto shape_y = std::vector<size_t>{rows};
  auto data_y = xt::adapt(raw_data_y, shape_y);
  std::cout << "Y shape " << data_y.shape() << std::endl;

  // generate new data
  xt::xarray<DType> new_x = xt::linspace<DType>(0.f, raw_data_x.back(), 2000);

  // straight line
  auto line_model = make_regression_model(data_x, data_y, 2);
  xt::xarray<DType> line_values = line_model(new_x);
  std::cout << "Line shape " << line_values.shape() << std::endl;

  // poly line
  auto poly_model = make_regression_model(data_x, data_y, 12);
  xt::xarray<DType> poly_line_values = poly_model(new_x);
  std::cout << "Poly line shape " << poly_line_values.shape() << std::endl;

  // create adaptors with STL like interfaces
  auto x_coord = xt::view(new_x, xt::all());
  auto line = xt::view(line_values, xt::all());
  auto polyline = xt::view(poly_line_values, xt::all());

  // plot the data we read and approximate
  plotcpp::Plot plt(true);
  plt.SetTerminal("qt");
  plt.SetTitle("Web traffic over the last month");
  plt.SetXLabel("Time");
  plt.SetYLabel("Hits/hour");
  plt.SetAutoscale();
  plt.GnuplotCommand("set grid");

  auto minmax = xt::eval(xt::minmax(data_x));
  auto time_range = minmax[0][1] - minmax[0][0];
  auto tic_size = 7 * 24;
  auto time_tics = time_range / tic_size;
  plt.SetXRange(-tic_size / 2, minmax[0][1] + tic_size / 2);

  plotcpp::Plot::Tics xtics;
  for (size_t t = 0; t < time_tics; ++t) {
    xtics.push_back({"week " + std::to_string(t), t * tic_size});
  }
  plt.SetXTics(xtics);

  plt.Draw2D(plotcpp::Points(data_x.begin(), data_x.end(), data_y.begin(),
                             "points", "lc rgb 'black' pt 1"),
             plotcpp::Lines(x_coord.begin(), x_coord.end(), line.begin(),
                            "line approx", "lc rgb 'red' lw 2"),
             plotcpp::Lines(x_coord.begin(), x_coord.end(), polyline.begin(),
                            "poly line approx", "lc rgb 'green' lw 2"));
  plt.Flush();

  return 0;
}
