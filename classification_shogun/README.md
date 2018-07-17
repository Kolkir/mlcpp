## Classification with Shogun machine learning library

[Shogun](http://www.shogun-toolbox.org/) is an open-source machine learning library that offers a wide range of machine learning algorithms. From my point of view it's not very popular among professionals, but it have a lot of fans among enthusiasts and students. Library offers unified API for algorithms, so they can be easily managed, it somehow looks like to ``scikit-learn`` approach. There is a set of [examples](http://www.shogun-toolbox.org/examples/latest/index.html) which can help you in learning of the library, but holistic documentation is missed.

In this article I will show how to use this library for solving a classification problem, and describe some not obvious details. I've used [Iris](https://www.kaggle.com/uciml/iris) dataset in this example, so loading data will depend on on it's format.

0. **Library installation**

    The easiest way to install the library is to compile it from sources, build scripts can be generated with CMake. Also I recommend to compile debug version too, because of lack of details in the documentation, it have sense to see what are happened in a debugger. Some Linux distributions offer complete package which can be installed directly to the system.

1. **Memory management**

      Most of data-structures in Shogun use reference counting approach to manage memory and resources, there are two base classes from which most of types are derived:
    * CSGObject - objects of this type can't be created on stack
    * SGReferencedData - objects of this type can be created on stack, and they usually use reference counting only for internal data.

    Library provides a kind on smart pointer to manage CSGObject type objects life-time, it called ``Some`` and you can use two functions to wrap objects in this smart pointer ``some`` and ``wrap``, here are examples how to use them:
    ```cpp
    // some - hides a new operator in the same way as std::make_shared does
    auto data_file = shogun::some<shogun::CCSVFile>(file_name);

    // wrap - automates creating of smart pointer object,
    // it can be useful in case of complex constructors
    auto labels = shogun::wrap(new shogun::CMulticlassLabels());

    // direct creation of the smart pointer
    auto labels = shogun::Some<shogun::CMulticlassLabels>(new shogun::CMulticlassLabels());
    ```

    You can't use objects of CSGObject type with c++ smart pointers because it uses internal implementation of reference counter.

    SGReferencedData type objects can be created on stack, but you should pay attention on how they are copied, because usually they don't support deep copying through the assignment operator and you should call a ``clone`` method if you need a real copy of object. Example of such types are SGMatrix and SGString.

2. **Loading data**

    First of all we need load data in SGMatrix object to be able to start processing. Library provides CVS file reader, which can be used for this task. This is an example how to do it:

    ```cpp
      auto data_file = shogun::some<shogun::CCSVFile>(file_name.c_str());

      // skip columns labels row
      data_file->set_lines_to_skip(1);

      // Load data from CSV to matrix
      Matrix data;
      data.load(data_file);
    ```
    Shogun CSV reader transposes data after read it, this issue can be solved by configuring reader with such call:
    ```cpp
    data_file->set_transpose(true);
    ```
    But pay attention, because on some files CSV reader ``crashed`` if this setting is enabled (Iris dataset is one of them). Also Shogun API treat matrix columns as samples and rows as features, so sometimes such type of automatic transposing can be useful.

    After we loaded the data to the matrix object we need to create a features and labels objects compatible with algorithms API in Shogun, you can't use SGMatrix with algorithms directly.

    Lets break Iris dataset matrix in two sets - samples+features and labels:
    ```cpp
    Matrix data;
    ...
    Matrix::transpose_matrix(data.matrix, data.num_rows, data.num_cols);
    // Exclude classification info from the data
    Matrix data_no_class = data.submatrix(0, data.num_cols - 1);  // make a view
    data_no_class = data_no_class.clone();  // copy exact data
    // Transpose matrix because shogun algorithms expect that samples are in columns
    Matrix::transpose_matrix(data_no_class.matrix, data_no_class.num_rows,
                             data_no_class.num_cols);
    auto features = shogun::some<shogun::CDenseFeatures<DType>>(data_no_class);
    auto labels =
        shogun::wrap(new shogun::CMulticlassLabels(features->get_num_vectors()));
    // data in the file is ordered by classes, so we can label it consequentially
    for (int i = 0; i < labels->get_num_labels() / 3; ++i) {
      labels->set_int_label(i, 0);
      labels->set_int_label(i + 50, 1);
      labels->set_int_label(i + 100, 2);
    }
    ```
    Look at ``submatrix`` method which I used to create a slice(view) of data in matrix, but it can be used only for columns range. Also I used a ``clone`` method to create a deep copy of data slice.

3. **Pre-processing data**

    Usually before passing a data to some machine learning algorithm it make sense to do some pre-processing like shuffling, scaling, dimensions reduction ... .

     Shogun provides useful functionality called `indices subsets` which can be used with features and labels to specify order of a data or define a reduced subset of data without copying or duplication. You can define indices subset with object of ``SGVector`` type.

    Example of shuffling a data with the indices subset:
    ```cpp
    shogun::SGVector<index_t> indices(labels->get_num_labels());
    indices.range_fill(); // values from 0 to indices vector length
    shogun::CMath::permute(indices);
    labels->add_subset(indices);
    features->add_subset(indices);
    ```
    To perform data scaling we have to create an object of a special type ``CRescaleFeatures``, initialize it with the data (to calculate mean and/or standard deviation), and then apply it to a data we want to scale:
    ```cpp
    auto scaler = shogun::wrap(new shogun::CRescaleFeatures());
    scaler->init(features);
    scaler->apply_to_feature_matrix(features);
    ```
    ``CRescaleFeatures`` can be applied only to the type derived from ``SGFeatures``, so you can't use it with a matrix directly, but it will modify the matrix you used to create features object.

    The same approach is used to perform dimensions reduction with PCA algorithm:
    ```cpp
    auto pca = shogun::wrap(new shogun::CPCA());
    pca->set_target_dim(2);  // before calling init method
    pca->init(features);
    pca->apply_to_feature_matrix(features);
    ```
    From this samples you can see that library uses unified API in different algorithms, ``init`` and ``apply_to_feature_matrix`` method are common for this kind of algorithms, so when you will try another one you can start with them.

4. **SVM**

     Lets configure and train SVM classifier, first of all we need initialize required parameters, and pass them to the classifier object:
    ```cpp
    auto svm = shogun::some<shogun::CMulticlassLibSVM>();
    auto kernel = shogun::wrap(new shogun::CGaussianKernel(5));
    svm->set_kernel(kernel);
    svm->set_C(1);
    svm->set_epsilon(0.00001);
    ```

    Shogun library provides classes to perform automatic cross validation during training, also there is a class for configuring grid search for models parameters. In this example I'm showing only cross validation:
    ```cpp
    // it means that data will be splitted in a training and a validation sets
    const int num_subsets = 2;

    // this class splits data in the equal ranges, also there are other splitting strategies in shogun
    auto splitting_strategy = shogun::some<shogun::CCrossValidationSplitting>(
       labels, num_subsets);

    auto evaluation_criterium = shogun::wrap(new shogun::CMulticlassAccuracy());

    auto cross = shogun::some<shogun::CCrossValidation>(
       svm, features, labels, splitting_strategy,
       evaluation_criterium);
    cross->set_num_runs(1);
    cross->set_autolock(false); // If true, machine will tried to be locked before evaluation
    ```
    Before actual training we need to setup special observer object for cross validation to be able to get machine(algorithm) parameters after the training. Training and validation processes can be launched with ``evaluate`` method:
    ```cpp
    auto obs = shogun::some<shogun::CParameterObserverCV>(true);
    cross->subscribe_to_parameters(obs);
    auto result =
       shogun::CCrossValidationResult::obtain_from_generic(cross->evaluate());
    std::cout << "Validation accuracy = " << result->get_mean() << std::endl;
    ```

    To get parameters we need to get a ``trained machine``, all algorithms which can be trained in Shogun are sub-classes of ``CMachine`` class. I took trained machine with ``get_trained_machine`` method from a fold object inside my observer. To manually evaluate algorithms on new data, machines usually have some kind of ``apply`` method and I used ``apply_multiclass`` in this case. After I got predictions I passed them to the ``evaluate`` method of object of ``CMulticlassAccuracy`` type:
    ```cpp
    auto machine = obs->get_observation(0)->get_fold(0)->get_trained_machine();
    auto svm_predict =
      shogun::wrap(machine->apply_multiclass(features));
      auto mult_accuracy_eval = shogun::wrap(new shogun::CMulticlassAccuracy());
    auto accuracy = mult_accuracy_eval->evaluate(svm_predict, labels);
    std::cout << "accuracy = " << accuracy << std::endl;
    ```
5. **Random Forest**

    To show the unified API of Shogun library I will show how to use Random Forest algorithm for the same classification task. At first I created object of ``CRandomForest`` type and configured it with required parameters:
    ```cpp
    auto rand_forest = shogun::some<shogun::CRandomForest>(0, 10);
    auto vote = shogun::some<shogun::CMajorityVote>();
    rand_forest->set_combination_rule(vote);
    // say to algorithm that our features are continuous
    auto featureTypes =
       shogun::SGVector<bool>(features->get_num_features());
    shogun::SGVector<bool>::fill_vector(featureTypes.vector, featureTypes.size(), false);
    rand_forest->set_feature_types(featureTypes);
    rand_forest->set_labels(labels);
    rand_forest->set_machine_problem_type(shogun::EProblemType::PT_MULTICLASS);
    ```
    I've planned to use same cross-validation training process as for SVM classifier, but there was a data race error (a multi-threading error) in the version of Shogun I used. So I just used a ``train`` method for the random forest machine:
    ```cpp
    rand_forest->train(features);
    ```
    For evaluation I used ``apply_multiclass`` method in this case too. After I got predictions I passed them to the ``evaluate`` method of object of ``CMulticlassAccuracy`` type as in the previous case:
    ```cpp
    auto forest_predict = shogun::wrap(rand_forest->apply_multiclass(features));
    auto mult_accuracy_eval = shogun::wrap(new shogun::CMulticlassAccuracy());
    auto accuracy = mult_accuracy_eval->evaluate(svm_predict, labels);
    std::cout << "accuracy = " << accuracy << std::endl;
    ```
