// third party includes
#include <shogun/base/init.h>
#include <shogun/base/some.h>
#include <shogun/distance/Distance.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/evaluation/ROCEvaluation.h>
#include <shogun/features/CombinedFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/io/File.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGStringList.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/parameter_observers/ParameterObserverCV.h>
#include <shogun/machine/RandomForest.h>
#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/MulticlassLibSVM.h>
#include <shogun/preprocessor/PCA.h>
#include <shogun/preprocessor/RescaleFeatures.h>
#include <shogun/util/factory.h>

#include <omp.h>

// stl includes
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

const std::string train_data_file_name =
    "/home/kirill/development/dataset/iris/iris.csv";

// Namespace and type aliases
using DType = float64_t;  // Offten it is impossible to use 32 bit float because
                          // some of algotims are hardcoded for 64 bit float.
using Matrix = shogun::SGMatrix<DType>;
using Data = std::tuple<shogun::Some<shogun::CDenseFeatures<DType>>,
                        shogun::Some<shogun::CMulticlassLabels>>;

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

  Matrix::transpose_matrix(data.matrix, data.num_rows, data.num_cols);

  // Exclude classification info from data
  Matrix data_no_class = data.submatrix(0, data.num_cols - 1);  // make a view
  data_no_class = data_no_class.clone();  // copy exact data

  // Transpose because shogun algorithms expect that samples are in colums
  Matrix::transpose_matrix(data_no_class.matrix, data_no_class.num_rows,
                           data_no_class.num_cols);

  auto features = shogun::some<shogun::CDenseFeatures<DType>>(data_no_class);

  // features->get_feature_matrix().display_matrix();

  std::cout << "samples = " << features->get_num_vectors()
            << " features = " << features->get_num_features() << std::endl;

  auto labels =
      shogun::wrap(new shogun::CMulticlassLabels(features->get_num_vectors()));

  std::cout << "labels = " << labels->get_num_labels() << std::endl;

  for (int i = 0; i < labels->get_num_labels() / 3; ++i) {
    labels->set_int_label(i, 0);
    labels->set_int_label(i + 50, 1);
    labels->set_int_label(i + 100, 2);
  }

  // shuffle data
  //  shogun::SGVector<index_t> indices(labels->get_num_labels());
  //  indices.range_fill();
  //  shogun::CMath::permute(indices);
  //  labels->add_subset(indices);
  //  features->add_subset(indices);

  std::cout << "classes = " << labels->get_num_classes() << std::endl;

  return std::make_tuple(features, labels);
}

void show_accuracy(shogun::Some<shogun::CMulticlassLabels> prediction,
                   shogun::Some<shogun::CMulticlassLabels> truth) {
  auto mult_accuracy_eval = shogun::wrap(new shogun::CMulticlassAccuracy());
  auto accuracy = mult_accuracy_eval->evaluate(prediction, truth);
  std::cout << "accuracy = " << accuracy << std::endl;

  // show predictions
  for (int i = 0; i < truth->get_num_labels(); ++i) {
    std::cout << "i = " << i << " predict = " << prediction->get_label(i)
              << " true = " << truth->get_label(i) << std::endl;
  }
}

void random_forest(Data train_data) {
  std::cout << "Train Random Forest ..." << std::endl;
  auto vote = shogun::some<shogun::CMajorityVote>();
  auto rand_forest = shogun::some<shogun::CRandomForest>(0, 10);
  rand_forest->set_combination_rule(vote);
  auto featureTypes =
      shogun::SGVector<bool>(std::get<0>(train_data)->get_num_features());
  shogun::SGVector<bool>::fill_vector(featureTypes.vector, featureTypes.size(),
                                      false);  // features are continuous
  rand_forest->set_feature_types(featureTypes);

  rand_forest->set_labels(std::get<1>(train_data));
  rand_forest->set_machine_problem_type(shogun::EProblemType::PT_MULTICLASS);

  if (!rand_forest->train(std::get<0>(train_data))) {
    std::cout << "Failed to train Random Forest\n";
  }

  // evaluate
  std::cout << "Evaluate ..." << std::endl;
  auto forest_predict =
      shogun::wrap(rand_forest->apply_multiclass(std::get<0>(train_data)));

  // estimate accuracy
  show_accuracy(forest_predict, std::get<1>(train_data));
}

void svm(Data train_data) {
  std::cout << "Train SVM ..." << std::endl;

  auto kernel = shogun::wrap(new shogun::CGaussianKernel(5));

  auto svm = shogun::some<shogun::CMulticlassLibSVM>();
  svm->set_kernel(kernel);
  svm->set_C(1);
  svm->set_epsilon(0.00001);

  const int num_subsets = 2;
  auto splitting_strategy = shogun::some<shogun::CCrossValidationSplitting>(
      std::get<1>(train_data), num_subsets);

  auto evaluation_criterium = shogun::wrap(new shogun::CMulticlassAccuracy());

  auto cross = shogun::some<shogun::CCrossValidation>(
      svm, std::get<0>(train_data), std::get<1>(train_data), splitting_strategy,
      evaluation_criterium);
  cross->set_num_runs(1);
  cross->set_autolock(false);

  auto obs = shogun::some<shogun::CParameterObserverCV>(true);
  cross->subscribe_to_parameters(obs);
  auto result =
      shogun::CCrossValidationResult::obtain_from_generic(cross->evaluate());
  std::cout << "Validation accuracy = " << result->get_mean() << std::endl;

  auto machine = obs->get_observation(0)->get_fold(0)->get_trained_machine();

  //  svm->set_labels(std::get<1>(train_data));
  //  svm->train(std::get<0>(train_data));
  //  auto machine = svm;

  // evaluate
  std::cout << "Evaluate ..." << std::endl;
  auto svm_predict =
      shogun::wrap(machine->apply_multiclass(std::get<0>(train_data)));

  // estimate accuracy
  show_accuracy(svm_predict, std::get<1>(train_data));
}

int main(int, char* []) {
  shogun::init_shogun_with_defaults();
  // shogun::sg_io->set_loglevel(shogun::MSG_INFO);
  shogun::sg_rand->set_seed(10);

  // load data
  std::cout << "Loading train data ..." << std::endl;
  auto train_data = load_data(train_data_file_name);

  // rescale
  std::cout << "Rescale data ..." << std::endl;
  auto scaler = shogun::wrap(new shogun::CRescaleFeatures());
  scaler->init(std::get<0>(train_data));
  scaler->apply_to_feature_matrix(std::get<0>(train_data));

  // PCA Dimension reduction
  std::cout << "PCA on data ..." << std::endl;
  auto pca = shogun::wrap(new shogun::CPCA());
  pca->set_target_dim(2);  // before init
  pca->init(std::get<0>(train_data));
  pca->apply_to_feature_matrix(std::get<0>(train_data));

  // Try models
  svm(train_data);
  random_forest(train_data);

  shogun::exit_shogun();
  return 0;
}
