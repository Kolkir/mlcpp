/* This sample is based on the Chapter 1 from book
 * "Building Machine Learning Systems with Python" by Willi Richert
 */

#include <csv.h>
#include <plot.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

// regular includes
#include <experimental/filesystem>
#include <iostream>
#include <string>
#include "../utils.h"

// Namespace and type aliases
namespace fs = std::experimental::filesystem;

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

  std::vector<float> x;
  std::vector<float> y;
  bool done = false;
  do {
    try {
      float x_v = 0, y_v = 0;
      done = !data_tsv.read_row(x_v, y_v);
      x.push_back(x_v);
      y.push_back(y_v);
    } catch (const io::error::no_digit& err) {
      std::cout << err.what() << std::endl;
    }
  } while (!done);

  // plot data we read
  plotcpp::Plot plt(true);
  plt.SetTerminal("qt");
  plt.SetTitle("Web traffic over the last month");
  plt.SetXLabel("Time");
  plt.SetYLabel("Hits/hour");
  plt.SetAutoscale();
  plt.GnuplotCommand("set grid");

  auto minmax = std::minmax_element(x.begin(), x.end());
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

  plt.Draw2D(plotcpp::Points(x.begin(), x.end(), y.begin(), "points"));
  plt.Flush();

  // fit polynom to our data
  std::vector<int> shape = {static_cast<int>(x.size()), 1};
  auto data_x = xt::adapt(x, shape);
  auto data_y = xt::adapt(y, shape);
  auto data = xt::stack(xt::xtuple(data_x, data_y), 1);
  std::cout << data.shape()[0] << " " << data.shape()[1];

  // xt::zeros<float> data{{3, 4}};

  return 0;
}
