## Classification with DLib machine learning library

[Dlib](http://dlib.net/) is an open source C++ framework containing various machine learning algorithms and many other complementary stuff which can be used for image processing, computer vision, linear algebra calculations and many other things. It has very good documentation and a lot of useful examples. 

In this article I will show how to use this library for solving a classification problem on [Iris](https://www.kaggle.com/uciml/iris) data set, so process of loading data will depend on its format.

0. **Library installation**

    I think the easiest way to get recent version of DLib is compile it from sources, there is a [tutorial](http://dlib.net/compile.html) on its site. Also you can find binary packages for Ubuntu, Debian and Arch Linux distributions. I've compiled library with ``Cuda`` support, but for doing this I've installed also ``gcc-6`` compiler for a host code and manually specified it in ``CMake`` parameters with ``CMAKE_CXX_COMPILER`` and ``CMAKE_C_COMPILER``, without such settings ``Cuda`` support can't be correctly configured. Also it have sense to compile DLib in debug mode with DLIB_ENABLE_ASSERTS CMake option enabled, to see inconvenience in data types, because it's often not obvious what value for template parameter is required to pass.

1. **Loading data**

    Dlib matrix class provides possibility to load in data from CSV formatted string. But initially I loaded data from network, and read it to a string:
    ```cpp
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
    ```
    To load data from string to DLib matrix, I specified exact size of the matrix, because library unable to determine it by itself. Also DLib internally uses double for a lot of calculations internally so in many cases it is impossible to use float as a base type for a Matrix (It will lead for compilation errors).
    ```cpp
    using DType = double;
    using Matrix = dlib::matrix<DType>;
    ...
    Matrix train_data(150, 5);
    std::stringstream ss(train_data_str);
    ss >> train_data;
    ```
    Next I split whole matrix to samples and labels, there are several methods to make a slice from matrix in DLib. Also pay attention that library use templates expressions for matrix calculations so final value for expression will be calculated in the assign operator. And next operations allocate new Matrix objects and copy data in them. 
    ```cpp
    // ----------- Extract labels - take last column
    Matrix labels = dlib::colm(train_data, 4);
    
    // ----------- Extract samples - take submatrix rectangle
    Matrix samples = dlib::subm_clipped(train_data, 0, 0, train_data.nr(),
                                        train_data.nc() - 1);
    ```
    
    Before pass data to DLib algorithms I converted samples matrix to c++ ``std::vector`` object, because its a library requirement. I've took each row with ``dlib::subm_clipped`` function, converted it to column vector and pushed to a vector.
    ```cpp
    using DataSet = std::pair<std::vector<Matrix>, std::vector<DType>>;
    ...
    DataSet dataset;
    for (long row = 0; row < samples.nr(); ++row) {
        dataset.first.push_back(dlib::reshape_to_column_vector(
        dlib::subm_clipped(samples, row, 0, 1, samples.nc())));
    }
    dataset.second.assign(labels.begin(), labels.end());
    ```

2. **Pre-processing data**
    
    Before going to classification algorithms I've shuffled data.
    
    ```cpp
    ...
    auto& [samples, labels] = dataset;
    dlib::randomize_samples(samples, labels);
    ```

    And extracted some set of samples and labels as test data.
    
    ```cpp
    std::vector<Matrix> test_data;
    std::ptrdiff_t split_point = 135;
    test_data.assign(samples.begin() + split_point, samples.end());
    samples.erase(samples.begin() + split_point, samples.end());
    std::vector<double> test_labels;
    test_labels.assign(labels.begin() + split_point, labels.end());
    labels.erase(labels.begin() + split_point, labels.end());
    ```
    
    After training some classification algorithm in DLib you will have a descision function, some of them possible to combine with normalization procedure so it can be a part of a model.

3. **SVM**

    I used this official [sample](http://dlib.net/model_selection_ex.cpp.html) as a base for my tutorial. At first I defined types for SVM classifier to be able to use it outside of a training function.
    
    ```cpp
    using svm_normalizer_type = dlib::vector_normalizer<Matrix>;
    using svm_kernel_type = dlib::radial_basis_kernel<Matrix>;
    using svm_ova_trainer = dlib::one_vs_all_trainer<dlib::any_trainer<Matrix>>;
    //---------- Classifier descision function 
    using svm_dec_funct_type = dlib::one_vs_all_decision_function<svm_ova_trainer>;   
    //--------- Classification function type which includes normalizer
    using svm_funct_type = dlib::normalized_function<svm_dec_funct_type, svm_normalizer_type>;
    ```
    
    Next step I made, was training a normalizer and data normalization. 
       
    ```cpp
    svm_normalizer_type normalizer;
    //---------- Let the normalizer learn the mean and standard deviation of the samples
    normalizer.train(samples);
    // --------- Here we normalize all the samples by subtracting their mean and dividing by their standard deviation.
    for (size_t i = 0; i < samples.size(); ++i) {
        samples[i] = normalizer(samples[i]);
    }
    ```
    Also there exists normalizer type combined with PCA method in Dlib, it allows to reduce data dimensions too.
    ```cpp
    using svm_normalizer_type = dlib::vector_normalizer_pca<Matrix>;
    svm_normalizer_type normalizer;
    normalizer.train(samples, 0.9); // configure how much dimensions will be left after PCA    
    ```
    Next I defined a function, that will do the Cross-Validation and return a number indicating how good a particular settings of gamma, c1, and c2 are.
    ```cpp
    auto cross_validation_score = [&](const DType gamma, const DType c1,
                                    const DType c2) {
        //-------- Make a RBF SVM trainer and tell it what the parameters are supposed to be.
        dlib::svm_c_trainer<svm_kernel_type> svm_trainer;
        svm_trainer.set_kernel(svm_kernel_type(gamma));
        svm_trainer.set_c_class1(c1);
        svm_trainer.set_c_class2(c2);
        
        //-------- Make one vs all trainer and add svm trainer to it
        svm_ova_trainer trainer;
        trainer.set_num_threads(4); // How much calculation threads to use
        trainer.set_trainer(svm_trainer);
        
        //-------- Perform 10-fold cross validation and return the results - confusion matrix.
        Matrix result =
            dlib::cross_validate_multiclass_trainer(trainer, samples, labels, 10);
         
        //-------- Return a number indicating how good the parameters are.  Bigger is better in this example.
        auto accuracy = sum(diag(result)) / sum(result);
        return accuracy;
    };
    ```
    I used global optimizer that searched the best parameters using the cross-validation function. It calls cross_validation_score() function 50 times with different settings and returns the best parameter settings it finds.
    ```cpp
    auto result = dlib::find_max_global(
      cross_validation_score,
      {1e-5, 1e-5,
       1e-5},  // lower bound constraints on gamma, c1, and c2, respectively
      {100, 1e6,
       1e6},   // upper bound constraints on gamma, c1, and c2, respectively
      dlib::max_function_calls(50));
    
    double best_gamma = result.x(0);
    double best_c1 = result.x(1);
    double best_c2 = result.x(2);
    ```
    
    After I got best values for parameters I repeated the training for the model with these parameters and saved results to a decision function.
    
    ```cpp
    dlib::svm_c_trainer<svm_kernel_type> svm_trainer;
    svm_trainer.set_kernel(svm_kernel_type(best_gamma));
    svm_trainer.set_c_class1(best_c1);
    svm_trainer.set_c_class2(best_c2);
    svm_ova_trainer trainer;
    trainer.set_num_threads(4);
    trainer.set_trainer(svm_trainer);
    
    svm_funct_type learned_function;
    learned_function.normalizer = normalizer;  // save normalization information
    //------- Perform the actual SVM training and save the results
    learned_function.function = trainer.train(samples, labels);
    ```
        
    For using this classifier model we can just call decision function with some sample data as argument, it will do normalization too.
    
    ```cpp   
    DType matches_num = 0;
    for (size_t i = 0; i < test_data.size(); ++i) {
        auto predicted_class = learned_function(test_data[i]);
        auto true_class = test_labels[i];
        if (predicted_class == true_class) {
          ++matches_num;
        }
    }
    auto accuracy = matches_num / test_data.size();
    ```

5. **Multilayer Perceptron**

    I planned to show another classification algorithm for DLib, but Decision trees and Random Forest algorithms are missed in DLib, there are another ones based on SVM. So to show something different for classification I used functionality to define and train neural networks, and made simple Multilayer Perceptron for this task. I used this [sample](http://dlib.net/dnn_introduction_ex.cpp.html) as a base for this tutorial. 
    
    As first step I changed labels type from double to unsigned int, because it is an API requirement.
    
    ```cpp
    std::vector<unsigned long> labels;
    labels.assign(real_labels.begin(), real_labels.end());
    ```
    Next I did normalization of training data in the same way as I did for SVM. 
    
    To configure NN we need to define its structure as a type with nested template arguments. I started with ``dlib::loss_multiclass_log`` type for a loss function and finished with ``dlib::input<delib::matrix<DType>>`` type for an input layer. As you can see you should define NN structure in DLib in revers order.
    ```cpp
    using namespace dlib;
    using net_type = loss_multiclass_log<
      fc<3, relu<fc<10, relu<fc<5, input<matrix<DType>>>>>>>>;
    net_type net;
    ```
    After I got NN object, I configured trainer and performed training. This trainer also tests if progress is still being made and if it isn't then it will reduce learning rate by setting it to ``learning_rate * learning_rate_shrink_factor``. But, it will not reduce it below ``min_learning_rate``. Once this minimum learning rate is crossed the training will terminate.
    ```cpp
    dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.01);
    trainer.set_min_learning_rate(0.00001);
    trainer.set_mini_batch_size(8);
    trainer.be_verbose();
    trainer.train(samples, labels);
    net.clean(); // Clean some intermediate data which is not used for evaluation
    ```

    It is impossible to define a combined with normalizer decision function for NN so you should do normalization of input data for evaluation manually. Evaluation of NN also can be done as a simple function call, but its decision function takes usually a batch(vector) of samples as argument in opposite to SVM decision function which takes only one sample for a call.
    ```cpp    
    for (auto& ts : test_data)
        ts = normalizer(ts);
    
    auto predicted_labels = net(test_data);
    DType matches_num = 0;
    for (size_t i = 0; i < test_data.size(); ++i) {
        auto predicted_class = predicted_labels[i];
        auto true_class = test_labels[i];
        if (predicted_class == true_class) {
          ++matches_num;
        }
    }
    auto accuracy = matches_num / test_data.size();
    ```
