/* This sample is based on the Chapter 1 from book
 * "Building Machine Learning Systems with Python" by Willi Richert
 */

// third party includes
#include <csv.h>
#include <plot.h>
#include <Eigen/Dense>

// stl includes
#include <algorithm>
#include <experimental/filesystem>
#include <iostream>
#include <random>
#include <string>

// application includes
#include "../ioutils.h"
#include "../utils.h"

// Namespace and type aliases
namespace fs = std::experimental::filesystem;
typedef double DType;
using Matrix = Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic>;

auto standardize(const Matrix& v) {
  assert(v.cols() == 1);
  auto m = v.colwise().mean();
  auto n = v.rows();
  DType sd = std::sqrt((v.rowwise() - m).array().pow(2).sum() /
                       static_cast<DType>(n - 1));
  Matrix sv = (v.rowwise() - m) / sd;
  return std::make_tuple(sv, m(0, 0), sd);
}

auto generate_polynomial(const Matrix& x, size_t degree) {
  assert(x.cols() == 1);
  auto rows = x.rows();

  Matrix poly_x = Matrix::Zero(rows, degree);
  // fill additional column for simpler vectorization
  {
    auto xv = poly_x.block(0, 0, rows, 1);
    xv.setOnes();
  }
  // copy initial data
  {
    auto xv = poly_x.block(0, 1, rows, 1);
    xv = x;
  }
  // generate additional terms
  for (size_t i = 2; i < degree; ++i) {
    auto xv = poly_x.block(0, i, rows, 1);
    xv = x.array().pow(static_cast<DType>(i));
  }
  return poly_x;
}

auto bgd(const Matrix& x, const Matrix& y) {
  size_t batch_size = 8;
  size_t n_epochs = 1000;
  DType lr = 0.0015;

  auto rows = x.rows();
  auto cols = x.cols();

  size_t batches = rows / batch_size;  // some samples will be skipped
  Matrix b = Matrix::Zero(cols, 1);

  DType prev_cost = std::numeric_limits<DType>::max();
  for (size_t i = 0; i < n_epochs; ++i) {
    for (size_t bi = 0; bi < batches; ++bi) {
      auto s = bi * batch_size;
      auto batch_x = x.block(s, 0, batch_size, cols);
      auto batch_y = y.block(s, 0, batch_size, 1);

      auto yhat = batch_x * b;
      auto error = yhat - batch_y;

      auto grad =
          (batch_x.transpose() * error) / static_cast<DType>(batch_size);

      b = b - lr * grad;
    }

    DType cost = (y - x * b).array().pow(2.f).sum() / static_cast<DType>(rows);

    std::cout << "BGD iteration : " << i << " Cost = " << cost << std::endl;
    if (cost <= prev_cost)
      prev_cost = cost;
    else
      break;  // early stopping
  }
  return b;
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
  size_t seed = 3465467546;
  std::shuffle(raw_data_x.begin(), raw_data_x.end(),
               std::default_random_engine(seed));
  std::shuffle(raw_data_y.begin(), raw_data_y.end(),
               std::default_random_engine(seed));

  // map data to the tensor
  size_t rows = raw_data_x.size();
  const auto data_x = Eigen::Map<Matrix>(raw_data_x.data(), rows, 1);
  std::cout << "X shape " << data_x.rows() << ":" << data_x.cols() << std::endl;
  Matrix x;
  std::tie(x, std::ignore, std::ignore) = standardize(data_x);

  const auto data_y = Eigen::Map<Matrix>(raw_data_y.data(), rows, 1);
  std::cout << "Y shape " << data_y.rows() << ":" << data_y.cols() << std::endl;
  // I'm not  using structured binding, because hard to debug such values
  Matrix y;
  DType ym{0};
  DType ysd{0};
  std::tie(y, ym, ysd) = standardize(data_y);

  // Scale data to prevent float overflow in BGD
  size_t p_degree = 64;
  const DType scale = 0.6;
  x *= scale;
  y *= scale;

  // solve normal equation
  Matrix poly_x = generate_polynomial(x, p_degree);
  Matrix b_eq =
      (poly_x.transpose() * poly_x).ldlt().solve(poly_x.transpose() * y);
  auto cost =
      (y - poly_x * b_eq).array().pow(2.f).sum() / static_cast<DType>(rows);
  std::cout << "cost for normal equation solution : " << cost << std::endl;

  // optimize with BGD
  Matrix b = bgd(poly_x, y);

  // generate new data
  const size_t new_x_size = 500;
  std::vector<DType> x_coord(new_x_size);
  auto new_x = Eigen::Map<Matrix>(x_coord.data(), new_x_size, 1);
  new_x = Eigen::Matrix<DType, Eigen::Dynamic, 1>::LinSpaced(
      new_x_size, data_x.minCoeff(), data_x.maxCoeff());
  Matrix new_x_std;
  std::tie(new_x_std, std::ignore, std::ignore) = standardize(new_x);
  new_x_std *= scale;
  Matrix new_poly_x = generate_polynomial(new_x_std, p_degree);

  // make predictions
  std::vector<DType> polyline_eq(new_x_size);
  auto new_y_eq = Eigen::Map<Matrix>(polyline_eq.data(), new_x_size, 1);
  new_y_eq = new_poly_x * b_eq;

  std::vector<DType> polyline(new_x_size);
  auto new_y = Eigen::Map<Matrix>(polyline.data(), new_x_size, 1);
  new_y = new_poly_x * b;

  // restore scalenew_y
  new_y_eq /= scale;
  new_y_eq = (new_y_eq * ysd).array() + ym;

  new_y /= scale;
  new_y = (new_y * ysd).array() + ym;

  // plot the data we read and approximate
  plotcpp::Plot plt(true);
  plt.SetTerminal("qt");
  plt.SetTitle("Web traffic over the last month");
  plt.SetXLabel("Time");
  plt.SetYLabel("Hits/hour");
  plt.SetAutoscale();
  plt.GnuplotCommand("set grid");

  auto time_range = data_x.maxCoeff() - data_x.minCoeff();
  auto tic_size = 7 * 24;
  auto time_tics = time_range / tic_size;
  plt.SetXRange(-tic_size / 2, data_x.maxCoeff() + tic_size / 2);

  plotcpp::Plot::Tics xtics;
  for (size_t t = 0; t < time_tics; ++t) {
    xtics.push_back({"week " + std::to_string(t), t * tic_size});
  }
  plt.SetXTics(xtics);

  plt.Draw2D(plotcpp::Points(raw_data_x.begin(), raw_data_x.end(),
                             raw_data_y.begin(), "data", "lc rgb 'black' pt 1"),
             plotcpp::Lines(x_coord.begin(), x_coord.end(), polyline_eq.begin(),
                            "neq approx", "lc rgb 'red' lw 2"),
             plotcpp::Lines(x_coord.begin(), x_coord.end(), polyline.begin(),
                            "bgd approx", "lc rgb 'cyan' lw 2"));
  plt.Flush();

  return 0;
}
