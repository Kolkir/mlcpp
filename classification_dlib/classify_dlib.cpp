// third party includes
#include <dlib/matrix.h>
#include <dlib/svm.h>
#include <plot.h>

// stl includes
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <streambuf>

// application includes
#include "../ioutils.h"
#include "../utils.h"

// Namespace and type aliases
namespace fs = std::experimental::filesystem;
using Matrix = dlib::matrix<float>;

static const std::string train_data_url =
    "https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/"
    "data/iris.csv";

Matrix LoadData() {
  // ----------- Download the data
  const std::string data_path{"iris.csv"};
  if (!fs::exists(data_path)) {
    if (!utils::DownloadFile(train_data_url, data_path)) {
      std::cerr << "Unable to download the file " << train_data_url
                << std::endl;
      return {};
    }
  }
  // ----------- Load data to string
  std::ifstream data_file(data_path);
  std::string train_data_str((std::istreambuf_iterator<char>(data_file)),
                             std::istreambuf_iterator<char>());

  // ----------- Remove first line - columns labels
  train_data_str.erase(0, train_data_str.find_first_of("\n") + 1);

  // ----------- Replace string labels with ints
  train_data_str =
      std::regex_replace(train_data_str, std::regex("Iris-setosa"), "0");
  train_data_str =
      std::regex_replace(train_data_str, std::regex("Iris-versicolor"), "1");
  train_data_str =
      std::regex_replace(train_data_str, std::regex("Iris-virginica"), "2");

  // ----------- Load data to matrix
  Matrix m(150, 5);
  std::stringstream ss(train_data_str);
  ss >> m;
  return m;
}

int main(int, char* []) {
  try {
    Matrix train_data = LoadData();
    // ----------- Extract labels
    Matrix labels = dlib::colm(train_data, 4);
    // std::cout << dlib::csv << labels << std::endl;
    // ----------- Extract and transpose samples
    Matrix samples = dlib::trans(dlib::subm_clipped(
        train_data, 0, 0, train_data.nr(), train_data.nc() - 1));
    // std::cout << dlib::csv << samples << std::endl;
    // ----------- Pre-process data
    dlib::randomize_samples(samples, labels);

  } catch (const std::exception& err) {
    std::cout << "Program crashed : " << err.what() << std::endl;
  }
  return 0;
}
