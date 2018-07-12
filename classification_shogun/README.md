## Classification with Shogun machine learning library

0. **Library installation**

    The easiest way to install library is to compile it from sources, build scripts can be easily generated with CMake. Also I recommend to compile debug version too, because of lack of details in documentation in have sense to see what happened in debugger.

1. **Memory management**
    * CSGObject - no stack, Some, wrap, some
    * SGReferencedData - SGMatrix - for copy use clone()

2. **Loading data**
    ```cpp
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
    ```
    * Mistake in CSV parser
    * Transposed loader
    * Creating feature
    * Creating labels
    * Indices subsets
    * Shuffling data
3. **Pre-processing data**
    * Scaling
    * PCA Dimension reduction
4. **SVM**
    * Initializing SVM parameters
    * Configuring Cross-Validation process
    * Training and getting parameters
    * Evaluating accuracy
5. **Random Forest**
    * Initializing parameters
    * Multi-threading error - unable to use Cross-Validation
    * Training
    * Evaluating accuracy
