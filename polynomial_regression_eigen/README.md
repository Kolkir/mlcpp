# Polynomial regression with Eigen library tutorial

Hello, this is my third article about how to use modern C++ for solving machine learning problems. This time I will show how to make a model for polynomial regression problem described in previous [article](https://github.com/Kolkir/mlcpp/tree/master/polynomial_regression), with well known linear algebra library called [Eigen](eigen.tuxfamily.org).

Eigen was chosen because it is widely used and has a long history, it is highly optimized for CPU, and is a header only library. One of the famous project using it is [TensorFlow](https://www.tensorflow.org/).

Please look at previous [article](https://github.com/Kolkir/mlcpp/tree/master/polynomial_regression) to learn how to download and parse training data.

1. **Loading data to Eigen data-structures**

    There are several approaches to initialize matrices in Eigen library, couple of them I used in next code sections.
    ```cpp
    typedef float DType;
    using Matrix = Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic>;
    ...
    size_t rows = raw_data_x.size();
    const auto data_x = Eigen::Map<Matrix>(raw_data_x.data(), rows, 1);
    ...
    Matrix poly_x = Matrix::Zero(rows, cols);
    ...
    Matrix m(rows, cols);
    ```
    Here I mapped raw C array of floats to ``Eigen::Matrix`` with ``Eigen::Map`` function, used special initialization method ``Matrix::Zero`` to define zero matrix with predefined size, and defined matrix with uninitialized values.

    Please look how I defined ``Matrix`` type - it uses dynamic memory management. I used such strategy because dimensions of training and evaluation data is not known at initial state. Eigen also supports static memory management strategy for matrices which has better performance.

2. **Standardization**

    To be able to perform successful computations for regression analysis we need to [standardize](https://en.wikipedia.org/wiki/Feature_scaling#Standardization) our data.
    ```cpp
    auto standardize(const Matrix& v) {
      auto m = v.colwise().mean();
      auto n = v.rows();
      DType sd = std::sqrt((v.rowwise() - m).array().pow(2).sum() /
      static_cast<DType>(n - 1));
      Matrix sv = (v.rowwise() - m) / sd;
      return std::make_tuple(sv, m(0, 0), sd);
    }
    ...
    // standardize data
    Matrix x;
    std::tie(x, std::ignore, std::ignore) = standardize(data_x);
    Matrix y;
    DType ym{0};
    DType ysd{0};
    // mean and std will be used later for scale restoring
    std::tie(y, ym, ysd) = standardize(data_y);
    ```
    For this piece of code I used ``std::tie`` function to unpack tuple parameters instead of using structured binding, because it is hard to debug such values, ``gdb`` don't see actual variables names. Also you should pay attention on Matrix class methods ``colwise`` and ``rowwise`` they used to apply a reduction operation on each column or row and return a column or row vector with the corresponding values. Not very obvious issue is that some trivial operations can be found as methods of ``Eigen::Array``, see how I called ``pow`` method after converting result of subtraction to the array.

3. **Generating additional polynomial components**

    To be able to approximate the data with higher degree polynomial I wrote a function for generating additional terms.

    ```cpp
    auto generate_polynomial(const Matrix& x, size_t degree) {
      auto rows = x.rows();
      Matrix poly_x = Matrix::Zero(rows, degree);
      poly_x.block(0, 0, rows, 1).setOnes();
      poly_x.block(0, 1, rows, 1) = x;
      for (size_t i = 2; i < degree; ++i) {
        auto xv = poly_x.block(0, i, rows, 1);
        xv = x.array().pow(static_cast<DType>(i));
      }
      return poly_x;
    }
    ...
    Matrix poly_x = generate_polynomial(x, p_degree);
    ...
    ```
    New matrix for ``X`` will look like ``X[i] = [1, x, x^2, x^3, ..., x^n]``, where ``X[i]`` is a row. Here I used ``block`` method to define a rectangular part of matrix on which I perform some operations.

4. **Batch gradient descent implementation**

    Batch gradient descent can be implemented very easily with Eigen, I used ``block`` operations to define batches, all matrix operations are implemented with overloaded math operators, so they look very natural. Also library supports automatic broadcasting, you can see an example for a single number on lines where gradients are multiplied with learning rate or are divided by a batch size. broadcasting is supported for a vectors too, and can be very effectively combined with partial reduction operations like ``colwise`` or ``rowwise``.

    ```cpp
    auto bgd(const Matrix& x, const Matrix& y) {
      ...
      Matrix b = Matrix::Zero(cols, 1);
      for (size_t i = 0; i < n_epochs; ++i) {
        for (size_t bi = 0; bi < n_batches; ++bi) {
          auto s = bi * batch_size;
          auto batch_x = x.block(s, 0, batch_size, cols);
          auto batch_y = y.block(s, 0, batch_size, 1);
          auto yhat = batch_x * b;
          auto error = yhat - batch_y;
          auto grad =
              (batch_x.transpose() * error) / static_cast<DType>(batch_size);
          b = b - learning_rate * grad;
        }
      }
      ...
      return b;
    }
    ...
    Matrix b = bgd(poly_x, y);
    ...
    ```

5. **Generating new data for testing model predictions**

      Before evaluation of a model, I generated a bunch of new data to make the test looks more naturaly. As underlaying data storage for a new data I used ``std::vector`` which was used in ``Eigen::Map`` operation to define a new matrix. It was made to have a C++ compatible iterators for a plotting library I used.

      Here I used matrix reduction functions ``minCoeff`` and ``maxCoeff`` to retrieve min and max coeficients. And I used  ``LinSpaced`` function to generate a range of consequent elements, pay attention on a type which used for LinSpaced call - it is a vector not a matrix (see last template parameter).

      ```cpp
      ...
      const size_t new_x_size = 500;
      std::vector<DType> x_coord(new_x_size);
      auto new_x = Eigen::Map<Matrix>(x_coord.data(), new_x_size, 1);
      new_x = Eigen::Matrix<DType, Eigen::Dynamic, 1>::LinSpaced(
          new_x_size, data_x.minCoeff(), data_x.maxCoeff());
      ...
      ```

6. **Making predictions**

    Making prediction can be expressed and simple matrix multiplication, and here I also defined a result matrix with ``std::vector`` to use it easily with plotting library.

    ```cpp
    ...
    std::vector<DType> polyline(new_x_size);
    auto new_y = Eigen::Map<Matrix>(polyline.data(), new_x_size, 1);
    new_y = new_poly_x * b;
    ...
    ```

7. **Plotting**

    To plot data I wrote [plotcpp]() library which is a thin wrapper for ``gnuplot`` application, and can show plots immediately in a new window.

    ```cpp
    plotcpp::Plot plt(true);
    plt.SetTerminal("qt");
    plt.SetTitle("Web traffic over the last month");
    plt.SetXLabel("Time");
    plt.SetYLabel("Hits/hour");
    plt.SetAutoscale();
    plt.GnuplotCommand("set grid");
    ...
    plt.Draw2D(plotcpp::Points(raw_data_x.begin(), raw_data_x.end(),
                             raw_data_y.begin(), "data", "lc rgb 'black' pt 1"),
               plotcpp::Lines(x_coord.begin(), x_coord.end(), polyline.begin(),
                            "bgd approx", "lc rgb 'cyan' lw 2"));
    ...
    plt.flush();
    ```
    ![plots](plot.png)

8. **Conclusion**

    I found Eigen as the most useful library for linear algebra in C++. It has intuitive interfaces and implements modern C++ approaches, for example you can use ``std::move`` to eliminate matrix coping in some cases. Also it has great [documentation](https://eigen.tuxfamily.org/dox/) with examples and search engine. It supports Intel® Math Kernel Library (MKL), which provides highly optimized multi-threaded mathematical routines for x86-compatible architectures. And it can be used in CUDA kernels, but this is still an experimental feature.

You can find full source of this example on [GitHub](https://github.com/Kolkir/mlcpp).
