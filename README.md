# mlcpp
Set of examples of ML approaches implemented in C++ with different frameworks.

After cloning the source code please execute next commands to get all required third parties:

```
git submodule init
git submodule update
```
Each folder contains single example with own ``CMakeLists.txt`` file.

**Linear Algebra**

1. [Polynomial regression with XTensor](https://github.com/Kolkir/mlcpp/tree/master/polynomial_regression) ``BSD 3-Clause`` ``CPU``
2. [Polynomial regression with MShadow on GPU](https://github.com/Kolkir/mlcpp/tree/master/polynomial_regression_gpu) ``Apache License 2.0`` ``CPU\GPU``
3. planned [Eigen](http://eigen.tuxfamily.org/) ``Mozilla Public License 2.0`` ``CPU``

**Full featured frameworks**

1. planned [Shark-ML](http://image.diku.dk/shark/) [sources link](https://github.com/Shark-ML/Shark) ``LGPL`` ``CPU\GPU``
2. planned [mlpack](https://github.com/mlpack/mlpack) ``BSD 3-Clause, Mozilla Public License 2, Boost Software License 1.0`` ``CPU``
3. planned [shogun-toolbox](http://www.shogun-toolbox.org/) ``BSD 3-Clause``  ``CPU\GPU``

**Deep Learning**

1. planned [MXNet](https://mxnet.apache.org/) [sources link](https://github.com/apache/incubator-mxnet/tree/master/cpp-package) ``Apache License 2.0``  ``CPU\GPU``
2. planned [Caffe2](https://caffe2.ai/) [sources link](https://github.com/caffe2/caffe2) ``Apache License 2.0`` ``CPU\GPU``
3. planned [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn) ``BSD 3-Clause`` ``CPU\embedded``
