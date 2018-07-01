// third party includes
#include <shogun/base/init.h>
#include <shogun/base/some.h>
#include <shogun/distance/Distance.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/evaluation/ROCEvaluation.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/io/File.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGStringList.h>
#include <shogun/lib/SGVector.h>
#include <shogun/machine/RandomForest.h>
#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/MulticlassLibSVM.h>
#include <shogun/preprocessor/PCA.h>
#include <shogun/preprocessor/RescaleFeatures.h>
#include <shogun/util/factory.h>

// stl includes
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

// application includes
#include "../ioutils.h"

// Namespace and type aliases
using DType = float64_t;  // Offten it is impossible to use 32 bit float because
                          // some of algotims are hardcoded for 64 bit float.
using Matrix = shogun::SGMatrix<DType>;
using Data = std::tuple<shogun::Some<shogun::CDenseFeatures<DType>>,
                        shogun::Some<shogun::CMulticlassLabels>>;

// Dataset reference:
// https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data

const std::string train_data_file_name =
    "/home/kirill/development/dataset/kepler-labelled-time-series-data/"
    "exoTrain.csv";

const std::string test_data_file_name =
    "/home/kirill/development/dataset/kepler-labelled-time-series-data/"
    "exoTest.csv";

Data load_data(const std::string& file_name) {
  // We can't define objects of types nherited from CSGObject on stack, because
  // algorithms use reference counting for such types of objects, and take it as
  // input parameter by pointer.

  auto data_file = shogun::some<shogun::CCSVFile>(file_name.c_str());

  // skip lables row
  data_file->set_lines_to_skip(1);

  // Shogun csv reader transpose data from file - restore right order
  // commented because some time leads to crash ???
  // data_file->set_transpose(true);

  // Load data from CSV to matrix
  // Matrix object can be created on stack
  Matrix data;
  data.load(data_file);

  std::cout << "samples = " << data.num_cols << " features = " << data.num_rows
            << std::endl;

  Matrix::transpose_matrix(data.matrix, data.num_rows, data.num_cols);

  // Exclude classification info from data
  Matrix data_no_class = data.submatrix(1, data.num_cols - 1);  // make a view
  data_no_class = data_no_class.clone();  // copy exact data

  // Transpose because shogun algorithms expect that samples are in colums
  Matrix::transpose_matrix(data_no_class.matrix, data_no_class.num_rows,
                           data_no_class.num_cols);

  auto features = shogun::some<shogun::CDenseFeatures<DType>>(data_no_class);

  auto labels = shogun::wrap(new shogun::CMulticlassLabels(data.num_rows));
  for (int i = 0; i < data.num_rows; ++i) {
    labels->set_int_label(i, data(i, 0) < 2 ? 0 : 1);
  }

  return std::make_tuple(features, labels);
}

auto make_binary_labels(shogun::Some<shogun::CMulticlassLabels> labels,
                        int value) {
  auto bin_labels =
      shogun::some<shogun::CBinaryLabels>(labels->get_num_labels());
  for (int i = 0; i < bin_labels->get_num_labels(); ++i) {
    bin_labels->set_int_label(i, labels->get_int_label(i) == value ? 1 : -1);
  }
  return bin_labels;
}

void show_accuracy(shogun::Some<shogun::CMulticlassLabels> prediction,
                   shogun::Some<shogun::CMulticlassLabels> truth) {
  auto mult_accuracy_eval = shogun::wrap(new shogun::CMulticlassAccuracy());
  auto accuracy = mult_accuracy_eval->evaluate(prediction, truth);
  std::cout << "accuracy = " << accuracy;

  auto roc_eval = shogun::wrap(new shogun::CROCEvaluation());

  auto bin_predict = make_binary_labels(prediction, 1);
  auto bin_truth = make_binary_labels(truth, 1);

  roc_eval->evaluate(bin_predict, bin_truth);

  std::cout << " auc = " << roc_eval->get_auROC() << std::endl;

  // show predictions
  for (int i = 0; i < 15; ++i) {
    std::cout << "i = " << i << " predict = " << prediction->get_label(i)
              << " true = " << truth->get_label(i) << std::endl;
  }
}

void svm(const Data& train_data, const Data& test_data) {
  std::cout << "Train SVM ..." << std::endl;
  auto kernel = shogun::wrap(new shogun::CGaussianKernel());
  kernel->init(std::get<0>(train_data), std::get<0>(train_data));

  auto C = 10.0;
  auto epsilon = 0.01;
  auto svm = shogun::wrap(
      new shogun::CMulticlassLibSVM(C, kernel, std::get<1>(train_data)));
  svm->set_epsilon(epsilon);
  if (!svm->train()) {
    std::cout << "Failed to train SVM\n";
  }

  // evaluate
  std::cout << "Evaluate ..." << std::endl;
  auto svm_prediction =
      shogun::wrap(svm->apply_multiclass(std::get<0>(test_data)));

  // estimate accuracy
  show_accuracy(svm_prediction, std::get<1>(test_data));
}

void random_forest(const Data& train_data, const Data& test_data) {
  std::cout << "Train Random Forest ..." << std::endl;
  auto vote = shogun::some<shogun::CMajorityVote>();
  auto rand_forest = shogun::some<shogun::CRandomForest>(
      std::get<0>(train_data), std::get<1>(train_data), 100);
  rand_forest->set_combination_rule(vote);
  //  auto featureTypes =
  //      shogun::SGVector<bool>(std::get<0>(train_data)->get_num_features());
  //  shogun::SGVector<bool>::fill_vector(featureTypes.vector,
  //  featureTypes.size(),
  //                                      true);  // feature are continuous
  //  rand_forest->set_feature_types(featureTypes);
  if (!rand_forest->train()) {
    std::cout << "Failed to train Random Forest\n";
  }

  // evaluate
  std::cout << "Evaluate ..." << std::endl;
  auto forest_predict =
      shogun::wrap(rand_forest->apply_multiclass(std::get<0>(test_data)));

  // estimate accuracy
  show_accuracy(forest_predict, std::get<1>(test_data));
}

int main(int, char* []) {
  shogun::init_shogun_with_defaults();

  // load data
  std::cout << "Loading train data ..." << std::endl;
  auto train_data = load_data(train_data_file_name);
  std::cout << "Loading test data ..." << std::endl;
  auto test_data = load_data(test_data_file_name);

  // rescale
  std::cout << "Rescale data ..." << std::endl;
  auto scaler = shogun::wrap(new shogun::CRescaleFeatures());
  scaler->init(std::get<0>(train_data));
  scaler->apply_to_feature_matrix(std::get<0>(train_data));
  scaler->apply_to_feature_matrix(std::get<0>(test_data));

  // PCA Dimension reduction
  std::cout << "PCA on data ..." << std::endl;
  auto pca = shogun::wrap(new shogun::CPCA());
  pca->set_target_dim(10);  // before init
  pca->init(std::get<0>(train_data));
  pca->apply_to_feature_matrix(std::get<0>(train_data));
  pca->apply_to_feature_matrix(std::get<0>(test_data));

  // Try models
  // svm(train_data, test_data);
  random_forest(train_data, test_data);

  shogun::exit_shogun();
  return 0;
}
