## Classification with Shark-ML machine learning library

[Shark-ML](http://www.shark-ml.org/) is an open-source machine learning library, it's site tells next: "_It provides methods for linear and nonlinear optimization, kernel-based learning algorithms, neural networks, and various other machine learning techniques. It serves as a powerful toolbox for real world applications as well as for research. Shark works on Windows, MacOS X, and Linux. It comes with extensive documentation. Shark is licensed under the GNU Lesser General Public License._" I can confirm that it offers a wide range of machine learning algorithms together with nice documentation, tutorials and samples. So in this article I will show the basic API concepts, details can be easily found in official [documentation](http://www.shark-ml.org/sphinx_pages/build/html/rest_sources/tutorials/tutorials.html). 

In this article I will show how to use this library for solving a classification problem. I've used [Iris](https://www.kaggle.com/uciml/iris) dataset in this example, so loading data will depend on on it's format.

0. **Library installation**

    Library should be compiled from sources, build scripts can be generated with CMake. Details about available CMake options can be found in [documentation](http://www.shark-ml.org/sphinx_pages/build/html/rest_sources/installation.html). Also you should have installed [Boost](https://www.boost.org/), because library depends from it. I did not find any problems with compilation. Also I used ``CMAKE_INSTALL_PREFIX`` option to changes default installation path, it allows to clear library artifacts after experiments.  

2. **Loading data**
    
    In this sample I'm going to download a dataset from internet. There is function ``shark::download`` in the library, but the problem is that it doesn't support ``https`` protocol. So I used function I wrote using  ``curl`` library:
    ```cpp
    static const std::string train_data_url =
    "https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/data/iris.csv";
    ...
    const std::string data_path{"iris.csv"};
    if (!fs::exists(data_path)) {
        if (!utils::DownloadFile(train_data_url, data_path)) {
          std::cerr << "Unable to download the file " << train_data_url
                    << std::endl;
          return 1;
        }
    }
    ```
    After file is downloaded you can use ``shark::csvStringToData`` do load data in ``shark::ClassificationDataset`` type object. But before do it you have to prepare this concrete dataset - remove row with columns names and replace labels string names with numbers:
    
     ```cpp
    std::ifstream data_file(data_path);
    std::string train_data_str((std::istreambuf_iterator<char>(data_file)),
                             std::istreambuf_iterator<char>());
    
    // ----------- Remove first line - columns labels
    train_data_str.erase(0, train_data_str.find_first_of("\n") + 1);
    
    // ----------- Replace string labels with integers
    train_data_str =
      std::regex_replace(train_data_str, std::regex("Iris-setosa"), "0");
    train_data_str =
      std::regex_replace(train_data_str, std::regex("Iris-versicolor"), "1");
    train_data_str =
      std::regex_replace(train_data_str, std::regex("Iris-virginica"), "2");
    ```
    
    Now we are ready to create to create a dataset object, with ``shark::LAST_COLUMN`` parameter we tell function that last column is labels:
    
    ```cpp
    shark::ClassificationDataset train_data;
    shark::csvStringToData(train_data, train_data_str, shark::LA ST_COLUMN);
    ```

3. **Pre-processing data**
    
    Before training classifiers or using other ML algorithms usually it's a good idea to normalize and shuffle your training data. Shark-ML already have good [tutorial](http://www.shark-ml.org/sphinx_pages/build/html/rest_sources/tutorials/concepts/data/normalization.html) about normalization. At first I shuffled the data and cutoff the part for a test set:
    ```cpp
    train_data.shuffle();
    auto test_data = shark::splitAtElement(train_data, 120);
    ```
    Then I defined a normalizer object and trainer for him. It's a common approach in Shark-ML to separate type for algorithm and for his trainer:
    ```cpp
    bool remove_mean = true;
    shark::Normalizer<shark::RealVector> normalizer;
    shark::NormalizeComponentsUnitVariance<shark::RealVector>
        normalizing_trainer(remove_mean);
    normalizing_trainer.train(normalizer, train_data.inputs());
    ```
    After trainer learned mean and variance and configured normalizer, we can use it to transform our data:
    ```cpp
    train_data = shark::transformInputs(train_data, normalizer);
    ```
    But there are trainers without a ``train`` method, lets see a PCA dimension reduction example:
    ```cpp
    shark::PCA pca(data.inputs());
    shark::LinearModel<> enc;
    pca.encoder(enc, 2);
    shark::Data<shark::RealVector> encoded_data = enc(data.inputs());
    ```
    Here ``pca`` object took data for learning in constructor and configured model for a dimension reduction with ``encoder`` method. 

4. **SVM**

5. **Random Forest**

