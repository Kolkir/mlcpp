# mlcpp
Set of examples of ML approaches implemented in C++ with different frameworks.

After cloning the source code please execute next commands to get all required third parties:

```
git submodule init
git submodule update
```
Each folder contains single example with own ``CMakeLists.txt`` file.

**Linear Algebra**

|Article|Library|CPU|GPU|Library's license|
|-------|-------|---|---|-----------------|
|[Polynomial regression](https://github.com/Kolkir/mlcpp/tree/master/polynomial_regression)|[XTensor](https://github.com/QuantStack/xtensor)|+||BSD 3-Clause|
|[Polynomial regression](https://github.com/Kolkir/mlcpp/tree/master/polynomial_regression_gpu)|[MShadow](https://github.com/dmlc/mshadow)|+|+|Apache License 2.0|
|[Polynomial regression](https://github.com/Kolkir/mlcpp/tree/master/polynomial_regression_eigen)|[Eigen](http://eigen.tuxfamily.org/)|+|?|Mozilla Public License 2.0|

**Full featured frameworks**

|Article|Library|CPU|GPU|Library's license|
|-------|-------|---|---|-----------------|
|planned|[Shark-ML](http://image.diku.dk/shark/)|+|+|LGPL|
|planned|[mlpack](https://github.com/mlpack/mlpack)|+||BSD 3-Clause, Mozilla Public License 2, Boost Software License 1.0|
|planned|[shogun-toolbox](http://www.shogun-toolbox.org/)|+|+|BSD 3-Clause|

**Deep Learning**

|Article|Library|CPU|GPU|Library's license|
|-------|-------|---|---|-----------------|
|planned|[MXNet](https://mxnet.apache.org/) [(sources)](https://github.com/apache/incubator-mxnet/tree/master/cpp-package)|+|+|Apache License 2.0|
|planned|[Caffe2](https://caffe2.ai/) [(sources)](https://github.com/caffe2/caffe2)|+|+|Apache License 2.0|
|planned|[tiny-dnn](https://github.com/tiny-dnn/tiny-dnn)|+||BSD 3-Clause|
