// third party includes
#include <dlib/dnn.h>
#include <dlib/global_optimization.h>
#include <dlib/matrix.h>
#include <dlib/svm.h>
#include <dlib/svm_threaded.h>
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
using DType = double;  // DLib internally uses double - so in many cases it is
                       // impossible to use float
using Matrix = dlib::matrix<DType>;
using DataSet = std::pair<std::vector<Matrix>, std::vector<DType>>;
//  SVM types
using svm_kernel_type = dlib::radial_basis_kernel<Matrix>;
using svm_ova_trainer = dlib::one_vs_all_trainer<dlib::any_trainer<Matrix>>;
using svm_dec_funct_type = dlib::one_vs_all_decision_function<svm_ova_trainer>;
// using svm_normalizer_type = dlib::vector_normalizer_pca<Matrix>;
using svm_normalizer_type = dlib::vector_normalizer<Matrix>;
using svm_funct_type =
    dlib::normalized_function<svm_dec_funct_type, svm_normalizer_type>;

static const std::string train_data_url =
    "https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/"
    "data/iris.csv";

DataSet LoadData() {
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
  Matrix train_data(150, 5);
  std::stringstream ss(train_data_str);
  ss >> train_data;
  // ----------- Extract labels
  Matrix labels = dlib::colm(train_data, 4);
  // std::cout << dlib::csv << labels << std::endl;
  // ----------- Extract and transpose samples
  Matrix samples = dlib::subm_clipped(train_data, 0, 0, train_data.nr(),
                                      train_data.nc() - 1);
  // std::cout << dlib::csv << samples << std::endl;
  // convert to matrices to vectors, because algorithms require such data types
  DataSet ds;
  for (long row = 0; row < samples.nr(); ++row) {
    ds.first.push_back(dlib::reshape_to_column_vector(
        dlib::subm_clipped(samples, row, 0, 1, samples.nc())));
  }
  ds.second.assign(labels.begin(), labels.end());
  return ds;
}

svm_funct_type TrainSVMClassifier(DataSet dataset) {
  // based on http://dlib.net/model_selection_ex.cpp.html

  auto& [samples, labels] = dataset;
  // ----------- Pre-process data
  // Here we normalize all the samples by subtracting their mean and
  // dividing by their standard deviation.
  svm_normalizer_type normalizer;
  // Let the normalizer learn the mean and standard deviation of the samples
  // normalizer.train(samples, 0.9); // configure how much dimensions will be
  // left after PCA
  normalizer.train(samples);
  std::cout << "Dimension of sample vector = " << normalizer.out_vector_size()
            << " after PCA" << std::endl;
  // now normalize each sample
  for (size_t i = 0; i < samples.size(); ++i) {
    samples[i] = normalizer(samples[i]);
  }

  // ----------- Select best parameters for svm model

  //  Here we define a function, that will do the cross-validation and return
  //  a number indicating how good a particular setting of gamma, c1, and c2
  //  is.
  auto cross_validation_score = [&](const DType gamma, const DType c1,
                                    const DType c2) {
    // Make a RBF SVM trainer and tell it what the parameters are supposed to
    // be.
    dlib::svm_c_trainer<svm_kernel_type> svm_trainer;
    svm_trainer.set_kernel(svm_kernel_type(gamma));
    svm_trainer.set_c_class1(c1);
    svm_trainer.set_c_class2(c2);

    svm_ova_trainer trainer;
    trainer.set_num_threads(4);
    trainer.set_trainer(svm_trainer);

    // Perform 10-fold cross validation and then print and return the
    // results - confusion matrix.
    Matrix result =
        dlib::cross_validate_multiclass_trainer(trainer, samples, labels, 10);
    auto accuracy = sum(diag(result)) / sum(result);
    std::cout << "gamma: " << gamma << "  c1: " << c1 << "  c2: " << c2
              << "\ncross validation accuracy: " << accuracy
              << "\nconfusion matrix:\n"
              << dlib::csv << result << std::endl;

    // Now return a number indicating how good the parameters are.  Bigger is
    // better in this example.
    return accuracy;
  };

  // Call this global optimizer that will search for the best
  // parameters. It will call cross_validation_score() 50 times with different
  // settings and return the best parameter setting it finds.
  auto result = dlib::find_max_global(
      cross_validation_score,
      {1e-5, 1e-5,
       1e-5},  // lower bound constraints on gamma, c1, and c2, respectively
      {100, 1e6,
       1e6},  // upper bound constraints on gamma, c1, and c2, respectively
      dlib::max_function_calls(50));

  double best_gamma = result.x(0);
  double best_c1 = result.x(1);
  double best_c2 = result.x(2);

  std::cout << " best cross-validation score: " << result.y << std::endl;
  std::cout << " best gamma: " << best_gamma << "   best c1: " << best_c1
            << "    best c2: " << best_c2 << std::endl;

  // ---------- Create final SVM model

  dlib::svm_c_trainer<svm_kernel_type> svm_trainer;
  svm_trainer.set_kernel(svm_kernel_type(best_gamma));
  svm_trainer.set_c_class1(best_c1);
  svm_trainer.set_c_class2(best_c2);
  svm_ova_trainer trainer;
  trainer.set_num_threads(4);
  trainer.set_trainer(svm_trainer);

  svm_funct_type learned_function;
  learned_function.normalizer = normalizer;  // save normalization information
  learned_function.function = trainer.train(
      samples,
      labels);  // perform the actual SVM training and save the results
  return learned_function;
}

auto TrainNNClassifier(DataSet dataset) {
  // based on http://dlib.net/dnn_introduction_ex.cpp.html
  using namespace dlib;

  auto& [samples, real_labels] = dataset;
  std::vector<unsigned long> labels;
  labels.assign(real_labels.begin(), real_labels.end());
  // ----------- Pre-process data
  // Here we normalize all the samples by subtracting their mean and
  // dividing by their standard deviation.
  vector_normalizer<Matrix> normalizer;
  // let the normalizer learn the mean and standard deviation of the samples
  normalizer.train(samples);
  // now normalize each sample
  for (size_t i = 0; i < samples.size(); ++i) {
    samples[i] = normalizer(samples[i]);
  }

  using net_type = loss_multiclass_log<
      fc<3, relu<fc<10, relu<fc<5, input<matrix<DType>>>>>>>>;
  net_type net;
  dnn_trainer<net_type> trainer(net);
  trainer.set_learning_rate(0.01);
  trainer.set_min_learning_rate(0.00001);
  trainer.set_mini_batch_size(8);
  trainer.be_verbose();
  trainer.train(samples, labels);
  net.clean();
  return std::make_pair(net, normalizer);
}

// ---------- Evaluate model on samples
template <typename Classfier>
void TestSVMClassfier(const std::string& name,
                      const Classfier& classifier,
                      DataSet dataset) {
  auto& [test_data, test_labels] = dataset;
  DType matches_num = 0;
  for (size_t i = 0; i < test_data.size(); ++i) {
    auto predicted_class = classifier(test_data[i]);
    auto true_class = test_labels[i];
    if (predicted_class == true_class) {
      ++matches_num;
    }
  }
  std::cout << name << " test accuracy = " << matches_num / test_data.size()
            << std::endl;
}

template <typename Classfier>
void TestNNClassfier(const std::string& name,
                     Classfier& classifier,
                     DataSet dataset) {
  auto& [test_data, test_labels] = dataset;
  for (auto& ts : test_data)
    ts = classifier.second(ts);  // normalize

  auto predicted_labels = classifier.first(test_data);
  DType matches_num = 0;
  for (size_t i = 0; i < test_data.size(); ++i) {
    auto predicted_class = predicted_labels[i];
    auto true_class = test_labels[i];
    if (predicted_class == true_class) {
      ++matches_num;
    }
  }
  std::cout << name << " test accuracy = " << matches_num / test_data.size()
            << std::endl;
}

int main(int, char* []) {
  using namespace std::string_literals;
  try {
    auto [samples, labels] = LoadData();

    // ----------- Pre-process data
    // It have sense to compile DLib in debug mode with DLIB_ENABLE_ASSERTS
    // CMake option enabled, to see inconvinience in data types

    dlib::randomize_samples(samples, labels);

    // exctract some samples as test data
    std::vector<Matrix> test_data;
    std::ptrdiff_t split_point = 135;
    test_data.assign(samples.begin() + split_point, samples.end());
    samples.erase(samples.begin() + split_point, samples.end());
    std::vector<double> test_labels;
    test_labels.assign(labels.begin() + split_point, labels.end());
    labels.erase(labels.begin() + split_point, labels.end());

    // ----------- SVM
    auto svm_classifier = TrainSVMClassifier({samples, labels});
    TestSVMClassfier("SVM"s, svm_classifier, {test_data, test_labels});

    // Decision trees and Random Forest algorithms are missed in DLib

    // ----------- Neural Net
    auto nn_classifier = TrainNNClassifier({samples, labels});
    TestNNClassfier("NN"s, nn_classifier, {test_data, test_labels});

  } catch (const std::exception& err) {
    std::cout << "Program crashed : " << err.what() << std::endl;
  }
  return 0;
}
