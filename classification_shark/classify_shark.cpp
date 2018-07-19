// third party includes
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h>
#include <shark/Data/Csv.h>
#include <shark/Models/ConcatenatedModel.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Normalizer.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

// stl includes
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <streambuf>

// application includes
#include "../ioutils.h"
#include "../utils.h"

// Namespace and type aliases
namespace fs = std::experimental::filesystem;

static const std::string train_data_url =
    "https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/"
    "data/iris.csv";

shark::ClassificationDataset LoadData() {
  // ----------- Download the data
  // SharkML - https download is not working
  // auto train_data_str = shark::download(train_data_url);

  const std::string data_path{"iris.csv"};
  if (!fs::exists(data_path)) {
    if (!utils::DownloadFile(train_data_url, data_path)) {
      std::cerr << "Unable to download the file " << train_data_url
                << std::endl;
      return 1;
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

  // ----------- Load data to Dataset
  shark::ClassificationDataset train_data;
  shark::csvStringToData(train_data, train_data_str, shark::LAST_COLUMN);
  return train_data;
}

int main(int, char* []) {
  try {
    shark::ClassificationDataset train_data = LoadData();
    // ----------- Preprocess data
    // http://image.diku.dk/shark/sphinx_pages/build/html/rest_sources/tutorials/concepts/data/normalization.html
    train_data.shuffle();
    auto test_data = shark::splitAtElement(train_data, 120);

    bool remove_mean = true;
    shark::Normalizer<shark::RealVector> normalizer;
    shark::NormalizeComponentsUnitVariance<shark::RealVector>
        normalizing_trainer(remove_mean);
    normalizing_trainer.train(normalizer, train_data.inputs());
    train_data = shark::transformInputs(train_data, normalizer);

    // ----------- SVM classificatoin
    // https://github.com/Shark-ML/Shark/blob/master/examples/Supervised/McSvm.tpp
    double c{10.0};
    double gamma{0.5};
    shark::GaussianRbfKernel<> kernel(gamma);
    shark::KernelClassifier<shark::RealVector> svm;
    shark::ZeroOneLoss<unsigned int> loss;
    shark::CSvmTrainer<shark::RealVector> trainer(&kernel, c, false);
    trainer.setMcSvmType(shark::McSvm::OVA);  // one-versus-all
    trainer.train(svm, train_data);
    auto output = svm(train_data.inputs());
    auto train_error = loss.eval(train_data.labels(), output);
    std::cout << "svm train error = " << train_error << std::endl;
    // We can't use shark::ConcatenatedModel class because there is a different
    // return type in KernelClassifier model
    // shark::ConcatenatedModel<shark::RealVector> svm_model;
    // svm_model.add(&normalizer, true);
    // svm_model.add(&svm, true);
    output = svm(normalizer(test_data.inputs()));
    auto test_error = loss.eval(test_data.labels(), output);
    std::cout << "svm test error = " << test_error << std::endl;

    // ----------- Random Forest classificatoin
    // http://image.diku.dk/shark/sphinx_pages/build/html/rest_sources/tutorials/algorithms/rf.html
  } catch (const std::exception& err) {
    std::cout << "Program crashed : " << err.what() << std::endl;
  }
  return 0;
}
