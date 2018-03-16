/* This sample is based on the Chapter 1 from book
 * "Building Machine Learning Systems with Python" by Willi Richert
 */

#include <mshadow/tensor.h>

// regular includes
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include "../iotensor.h"
#include "../utils.h"

// Namespace and type aliases
namespace fs = std::experimental::filesystem;
namespace ms = mshadow;
using device = ms::cpu;

// tensor engine needed for CuBLAS
struct ScopedTensorEngine {
  ScopedTensorEngine() { ms::InitTensorEngine<device>(); }
  ~ScopedTensorEngine() { ms::ShutdownTensorEngine<device>(); }
};

int main() {
  ScopedTensorEngine tensorEngine;

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
  std::ifstream data_file(data_path);
  if (data_file) {
    // count samples
    auto n = static_cast<size_t>(
        std::count(std::istreambuf_iterator<char>(data_file),
                   std::istreambuf_iterator<char>(), '\n'));
    data_file.seekg(0, std::ios_base::beg);

    // allocate tensor
    ms::TensorContainer<device, 2> data(ms::Shape2(n, 2));

    // read samples
    std::string line_str;
    std::stringstream line_fmt_stream;
    for (size_t i = 0; i < n; ++i) {
      if (std::getline(data_file, line_str)) {
        line_fmt_stream << line_str;
        if (!(line_fmt_stream >> data[i][0]))
          data[i][0] = std::numeric_limits<ms::default_real_t>::quiet_NaN();
        if (!(line_fmt_stream >> data[i][1]))
          data[i][1] = std::numeric_limits<ms::default_real_t>::quiet_NaN();
        std::stringstream().swap(line_fmt_stream);
      } else {
        std::cerr << "Error parsing the file " << data_path << " line: " << i
                  << std::endl;
        return 1;
      }
    }

    // check data we read
    auto slice = data.Slice(0, 10);
    for (size_t y = 0; y < slice.shape_[0]; ++y) {
      std::cout << "[";
      for (size_t x = 0; x < slice.shape_[1]; ++x) {
        std::cout << slice[y][x] << ", ";
      }
      std::cout << "]\n";
    }

  } else {
    std::cerr << "Unable to read the file " << data_path << std::endl;
    return 1;
  }

  return 0;
}
