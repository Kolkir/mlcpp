# mlcpp
Set of examples of ML approaches implemented in C++ with [XTensor](https://github.com/QuantStack/xtensor) and [MShadow](https://github.com/dmlc/mshadow) libraries.

After cloning the source code please execute next commands to get all required third parties:

```
git submodule init
git submodule update
```
Each folder contains single example with own ``CMakeLists.txt`` file.

**Linear Math**

1. \+ [Polynomial regression with XTensor](https://github.com/Kolkir/mlcpp/tree/master/polynomial_regression)
2. \+ [Polynomial regression with MShadow on GPU](https://github.com/Kolkir/mlcpp/tree/master/polynomial_regression_gpu)

**Full featured frameworks**

1. \- [Shark-ML](http://image.diku.dk/shark/) [sources link](https://github.com/Shark-ML/Shark)
2. \- [mlpack](https://github.com/mlpack/mlpack)
3. \- [shogun-toolbox](http://www.shogun-toolbox.org/)

**Deep Learning**

1. \- [MXNet](https://mxnet.apache.org/)
2. \- [Caffe2](https://caffe2.ai/)
