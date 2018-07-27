// third party includes
#include <plot.h>
#include <shark/Algorithms/DirectSearch/GridSearch.h>
#include <shark/Algorithms/JaakkolaHeuristic.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h>
#include <shark/Algorithms/Trainers/PCA.h>
#include <shark/Algorithms/Trainers/RFTrainer.h>
#include <shark/Data/Csv.h>
#include <shark/Models/ConcatenatedModel.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Normalizer.h>
#include <shark/ObjectiveFunctions/CrossValidationError.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

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
#include "class_iterator.h"

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

// ----------- Show predictions

void ShowModel(const shark::ClassificationDataset& true_data,
               const shark::Data<unsigned int>& predictions) {
  shark::PCA pca(true_data.inputs());
  shark::LinearModel<> enc;
  pca.encoder(enc, 2);
  shark::Data<shark::RealVector> encoded_data = enc(true_data.inputs());
  auto& lables = true_data.labels();

  ClassIterator di_0_x(&encoded_data, &lables, 0, 0);
  ClassIterator di_0_y(&encoded_data, &lables, 0, 1);

  ClassIterator di_1_x(&encoded_data, &lables, 1, 0);
  ClassIterator di_1_y(&encoded_data, &lables, 1, 1);

  ClassIterator di_2_x(&encoded_data, &lables, 2, 0);
  ClassIterator di_2_y(&encoded_data, &lables, 2, 1);

  ClassIterator pdi_0_x(&encoded_data, &predictions, 0, 0);
  ClassIterator pdi_0_y(&encoded_data, &predictions, 0, 1);

  ClassIterator pdi_1_x(&encoded_data, &predictions, 1, 0);
  ClassIterator pdi_1_y(&encoded_data, &predictions, 1, 1);

  ClassIterator pdi_2_x(&encoded_data, &predictions, 2, 0);
  ClassIterator pdi_2_y(&encoded_data, &predictions, 2, 1);

  plotcpp::Plot plt(true);
  plt.SetTerminal("qt");
  plt.SetAutoscale();
  plt.GnuplotCommand("set grid");
  plt.Draw2D(plotcpp::Points(di_0_x, ClassIterator(), di_0_y, "class 0",
                             "lc rgb 'red' pt 4"),
             plotcpp::Points(di_1_x, ClassIterator(), di_1_y, "class 1",
                             "lc rgb 'green' pt 4"),
             plotcpp::Points(di_2_x, ClassIterator(), di_2_y, "class 2",
                             "lc rgb 'blue' pt 4"),
             plotcpp::Points(pdi_0_x, ClassIterator(), pdi_0_y, "predict 0",
                             "lc rgb 'red' pt 1"),
             plotcpp::Points(pdi_1_x, ClassIterator(), pdi_1_y, "predict 1",
                             "lc rgb 'green' pt 1"),
             plotcpp::Points(pdi_2_x, ClassIterator(), pdi_2_y, "predict 2",
                             "lc rgb 'blue' pt 1"));
  plt.Flush();
}

// ----------- Model evaluation
using Model = shark::AbstractModel<remora::vector<double, remora::cpu_tag>,
                                   unsigned int,
                                   remora::vector<double, remora::cpu_tag>>;
void EvaluateModel(const std::string& name,
                   const Model& model,
                   const shark::Normalizer<shark::RealVector>& normalizer,
                   const shark::ClassificationDataset& train_data,
                   const shark::ClassificationDataset& test_data) {
  shark::ZeroOneLoss<unsigned int> loss;
  auto output = model(train_data.inputs());
  auto train_error = loss.eval(train_data.labels(), output);
  std::cout << name << " train error = " << train_error << std::endl;

  ShowModel(train_data, output);

  output = model(normalizer(test_data.inputs()));
  auto test_error = loss.eval(test_data.labels(), output);
  std::cout << name << " test error = " << test_error << std::endl;
}

// ----------- SVM classificatoin
// https://github.com/Shark-ML/Shark/blob/master/examples/Supervised/McSvm.tpp
// http://www.shark-ml.org/sphinx_pages/build/html/rest_sources/tutorials/algorithms/svm.html
// http://www.shark-ml.org/sphinx_pages/build/html/rest_sources/tutorials/algorithms/svmModelSelection.html
struct SVMModel {
  SVMModel(double gamma, bool unconstrained) : kernel(gamma, unconstrained) {}
  // it was not obvious which objects life time you should take care
  shark::GaussianRbfKernel<> kernel;
  // template parameter is input type
  shark::KernelClassifier<shark::RealVector> model;
};
auto TrySVM(const shark::ClassificationDataset& train_data,
            const shark::CVFolds<shark::ClassificationDataset>& folds) {
  double c{1.0};
  double gamma{0.5};
  bool offset = true;
  bool unconstrained = true;

  auto svm = std::make_shared<SVMModel>(gamma, unconstrained);

  shark::CSvmTrainer<shark::RealVector> trainer(&svm->kernel, c, offset,
                                                unconstrained);
  trainer.setMcSvmType(shark::McSvm::OVA);  // one-versus-all

  shark::ZeroOneLoss<unsigned int> loss;

  // configure CV
  shark::CrossValidationError<shark::KernelClassifier<shark::RealVector>,
                              unsigned int>
      cv_error(folds, &trainer, &svm->model, &trainer, &loss);

  // estimate initial parameters values
  shark::JaakkolaHeuristic ja(train_data);
  double ljg = log(ja.gamma());

  // we have two hyperparameters so define the grid accordingly
  shark::GridSearch grid;
  std::vector<double> min(2);
  std::vector<double> max(2);
  std::vector<size_t> sections(2);
  // kernel parameter gamma
  min[0] = ljg - 4.;
  max[0] = ljg + 4;
  sections[0] = 9;
  // regularization parameter C
  min[1] = 0.0;
  max[1] = 10.0;
  sections[1] = 11;
  grid.configure(min, max, sections);
  grid.step(cv_error);

  trainer.setParameterVector(grid.solution().point);
  trainer.train(svm->model, train_data);
  return svm;
}

// ----------- Random Forest classification
// http://www.shark-ml.org/sphinx_pages/build/html/rest_sources/tutorials/algorithms/rf.html
auto TryRF(const shark::ClassificationDataset& train_data,
           const shark::CVFolds<shark::ClassificationDataset>& folds) {
  // template parameter is label type
  shark::RFTrainer<unsigned int> trainer;
  auto rf = std::make_shared<shark::RFClassifier<unsigned int>>();
  trainer.train(*rf, train_data);
  return rf;
}

int main(int, char* []) {
  try {
    shark::ClassificationDataset train_data = LoadData();
    // ----------- Preprocess data
    // www.shark-ml.org/sphinx_pages/build/html/rest_sources/tutorials/concepts/data/normalization.html
    train_data.shuffle();
    auto test_data = shark::splitAtElement(train_data, 120);

    bool remove_mean = true;
    shark::Normalizer<shark::RealVector> normalizer;
    shark::NormalizeComponentsUnitVariance<shark::RealVector>
        normalizing_trainer(remove_mean);
    normalizing_trainer.train(normalizer, train_data.inputs());
    train_data = shark::transformInputs(train_data, normalizer);

    // ----------- Split data in folds for CV training
    const unsigned int k = 5;  // number of folds
    shark::CVFolds<shark::ClassificationDataset> folds =
        shark::createCVSameSizeBalanced(train_data, k);

    // ----------- SVM classificatoin
    auto svm_model = TrySVM(train_data, folds);
    EvaluateModel("svm", svm_model->model, normalizer, train_data, test_data);

    // ----------- Random Forest classificatoin
    auto rf_model = TryRF(train_data, folds);
    EvaluateModel("random forest", *rf_model, normalizer, train_data,
                  test_data);

  } catch (const std::exception& err) {
    std::cout << "Program crashed : " << err.what() << std::endl;
  }
  return 0;
}
