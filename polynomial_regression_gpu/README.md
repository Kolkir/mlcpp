# Polynomial regression with MShadow library tutorial

Hello, this is my second article about how to use modern C++ for solving machine learning problems. This time I will show how to make a model for polynomial regression problem described in previous [article](https://github.com/Kolkir/mlcpp/tree/master/polynomial_regression), but now with another library which allows you to use your GPU easily.

For this tutorial I chose [MShadow](https://github.com/dmlc/mshadow) library, you can find documentation for it [here](https://github.com/dmlc/mshadow/tree/master/doc). This library was chosen because it is actively developed now, and used as a basis for one of a wide used deep learning framework [MXNet](https://mxnet.incubator.apache.org/). Also it is a header only library with minimal dependencies, so it's integration is not hard at all.

Please look at previous [article](https://github.com/Kolkir/mlcpp/tree/master/polynomial_regression) to learn how to download and parse training data. 

You have pay attention on how sources for this tutorial are compiled, I used CUDA compiler for them, please look at corresponding CMakeLists.txt file for details. Also you should have installed ``gcc-6`` as host compiler for ``CUDA 9``.

0. **Preparations**
	MShadow library use special routines to initialize and shutdown itself,  I wrote a simple class to use them in RAII manner:
	``` cpp
	#include <mshadow/tensor.h>
	...
	namespace ms = mshadow;
	typedef float DType;
	...
	template <typename Device>
	struct ScopedTensorEngine {
	  ScopedTensorEngine() { ms::InitTensorEngine<Device>(); }
	  ~ScopedTensorEngine() { ms::ShutdownTensorEngine<Device>(); }
	  ScopedTensorEngine(const ScopedTensorEngine&) = delete;
	  ScopedTensorEngine& operator=(const ScopedTensorEngine&) = delete;
	};
	```
	Device template parameter c
	
2. **Loading data to MShadow datastructures**

    
3. **Standardization**

   
4. **Generating new data for testing model predictions**

  
5. **Batch gradient descent implementation**

   
6. **Generating additional polynomial components**

 
7. **Creating general regression model**

    
8. **Making predictions**

   
9. **Plot results**

    
You can find full source of this example on [GitHub](https://github.com/Kolkir/mlcpp).
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE2MjU5Mjk2MTEsMTE3NzE4NjY2OSwxOT
k5NzAyNzYyLDE1Mjk2NDI2NDcsLTE3MzY0ODcyNDgsLTE3Mjk5
NzY2NTddfQ==
-->