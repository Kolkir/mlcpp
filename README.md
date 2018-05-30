# mlcpp
Set of examples of ML approaches implemented in C++ with [XTensor](https://github.com/QuantStack/xtensor) and [MShadow](https://github.com/dmlc/mshadow) libraries.

After cloning the source code please execute next commands to get all required third parties:

```
git submodule init
git submodule update
```
Each folder contains single example with own ``CMakeLists.txt`` file.

Content:
1. [Polynomial regression with XTensor](https://github.com/Kolkir/mlcpp/tree/master/polynomial_regression)
2. [Polynomial regression with MShadow on GPU](https://github.com/Kolkir/mlcpp/tree/master/polynomial_regression_gpu)
