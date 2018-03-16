/* This sample is based on the Chapter 1 from book
 * "Building Machine Learning Systems with Python" by Willi Richert
 */

#include "../utils.h"

#include <experimental/filesystem>
#include <iostream>
#include <string>

namespace fs = std::experimental::filesystem;

static const std::string data_url{
    R"(https://raw.githubusercontent.com/luispedro/BuildingMachineLearningSystemsWithPython/master/ch01/data/web_traffic.tsv)"};
static const std::string destination_path{"web_traffic.tsv"};

int main() {
  // Download the data
  if (!fs::exists(destination_path)) {
    if (!utils::DownloadFile(data_url, destination_path)) {
      std::cerr << "Unable to download the file " << data_url << std::endl;
      return 1;
    }
  }

  return 0;
}
