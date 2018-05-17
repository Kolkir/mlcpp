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
	Device template parameter can be ``ms::cpu`` or ``ms::gpu``, I will instantiate ``ScopedTensorEngine`` for both types, because I need to pass data from host side to GPU (but it is allowed to use only ``ms::cpu`` for all computations, and code will remain the same):
	```cpp
	ScopedTensorEngine<ms::cpu> tensorEngineCpu;
	ScopedTensorEngine<ms::gpu> tensorEngineGpu;
	```
	Next I defined a variable which will represent a CUDA stream. A CUDA Stream is a sequence of operations that are performed in order on the GPU device. Streams can be run in independent concurrent in-order queues of execution, and operations in different streams can be interleaved and overlapped. This variable is necessary for using other MShadow abstractions. 
	```cpp
	...
	using GpuStream = ms::Stream<ms::gpu>;
    using GpuStreamPtr = std::unique_ptr<GpuStream, void (*)(GpuStream*)>;
    ...
    GpuStreamPtr computeStream(ms::NewStream<ms::gpu>(true, false, -1), 
        [](GpuStream* s) { ms::DeleteStream(s); });
	```
	C++ smart pointer with custom deleter can be very useful for C style interfaces.  
	
1. **Loading data to MShadow datastructures**
	There are several approaches to initialize tensors data structures in MShadow library, two of them I used next code section.
	```cpp
	  auto rows = raw_data_x.size();
	  ms::Tensor<ms::cpu, 2, DType> host_y(raw_data_y.data(), ms::Shape2(rows, 1));
	  ms::TensorContainer<ms::gpu, 2, DType> gpu_y(host_y.shape_);
	  gpu_y.set_stream(computeStream.get());
	  ms::Copy(gpu_y, host_y, computeStream.get());
	```
    When I initialize ``host_y`` variable I provide pointer to raw data array in constructor, so in this case tensor will work as wrapper around raw array. It's very useful technique to work with host data to eliminate unnecessary copying.  Next I used ``ms::TensorContainer`` type which implements RAII idiom for ``ms::Tensor``, it will allocate required amount of memory and free it in a destructor.  I found it useful for managing GPU data, but library authors recommend it mostly for intermediate calculations results. Also pay attention on how CUDA stream is used, for ``gpu_y`` initialization and during copy operation. 
    
3. **Standardization**
To be able to perform successful computations for regression analysis we need to [standardize](https://en.wikipedia.org/wiki/Feature_scaling#Standardization) our data. Also because we need to preallocate several  intermediate tensors for calculations and to reuse a code I implemented standardization procedure as separate class.
	```cpp
	class Standardizer {
	 public:
	  using T = ms::TensorContainer<ms::gpu, 2, DType>;

	  Standardizer(GpuStream* computeStream, size_t rows)
	      : min(ms::Shape1(1)),
	        max(ms::Shape1(1)),
	        mean(ms::Shape1(1)),
	        temp(ms::Shape2(rows, 1)),
	        sd(ms::Shape1(1)),
	        rows(rows) {
	    min.set_stream(computeStream);
	    max.set_stream(computeStream);
	    mean.set_stream(computeStream);
	    sd.set_stream(computeStream);
	    temp.set_stream(computeStream);
	  }
	  ~Standardizer() {}
	  Standardizer(const Standardizer&) = delete;
	  Standardizer& operator=(const Standardizer&) = delete;

	  void standardize(T& vec, GpuStream* computeStream) {
	    mean = ms::expr::sumall_except_dim<1>(vec);
	    mean /= static_cast<DType>(rows);
	    temp = ms::expr::F<Pow>(vec - ms::expr::broadcast<1>(mean, temp.shape_), 2);
	    sd = ms::expr::sumall_except_dim<1>(temp);
	    sd = ms::expr::F<Sqrt>(sd) / static_cast<DType>(rows - 1);
	    temp = (vec - ms::expr::broadcast<1>(mean, temp.shape_)) /
	           ms::expr::broadcast<1>(sd, temp.shape_);

	    // scale to [-1, 1] range
	    min =
	        ms::expr::ReduceTo1DExp<T, DType, ms::red::minimum,
	                                ms::expr::ExpInfo<T>::kDim - 1>(temp, DType(1));
	    max =
	        ms::expr::ReduceTo1DExp<T, DType, ms::red::maximum,
	                                ms::expr::ExpInfo<T>::kDim - 1>(temp, DType(1));

	    temp = (temp - ms::expr::broadcast<1>(min, temp.shape_)) /
	           ms::expr::broadcast<1>(max - min, temp.shape_);

	    temp = (temp * 2.f) - 1.f;

	    ms::Copy(vec, temp, computeStream);
	  }
	  auto get_moments(GpuStream* computeStream) {
	    ms::TensorContainer<ms::cpu, 1, DType> value(ms::Shape1(1));
	    ms::Copy(value, min, computeStream);
	    DType v_min = value[0];
	    ms::Copy(value, max, computeStream);
	    DType v_max = value[0];
	    ms::Copy(value, mean, computeStream);
	    DType v_mean = value[0];
	    ms::Copy(value, sd, computeStream);
	    DType v_sd = value[0];
	    return std::vector<DType>{v_min, v_max, v_mean, v_sd};
	  }
	 private:
	  ms::TensorContainer<ms::gpu, 1, DType> min;
	  ms::TensorContainer<ms::gpu, 1, DType> max;
	  ms::TensorContainer<ms::gpu, 1, DType> mean;
	  ms::TensorContainer<ms::gpu, 1, DType> sd;
	  ms::TensorContainer<ms::gpu, 2, DType> temp;
	  size_t rows;
	};
	``` 
	The interesting moments here are :
	1.  ``ms::expr::broadcast`` function which make possible to define element wise operations for tensors with single value, for example subtraction one number from each tensor element. There is a dynamic broadcasting in this library, but to use it you need actual value (it doesn't work for expressions), so in some cases it will require earlier expression evaluation which can hurt performance.
	2.  ``ms::expr::sumall_except_dim`` function which calculate sum of elements along not specified tensor dimension. 
	3.  ``ms::expr::F`` custom user specified operation on tensor elements, I used power and square root operations:
		```cpp
		struct Pow {
		  MSHADOW_XINLINE static float Map(float x, float y) { return pow(x, y); }
		};
		
		struct Sqrt {
		  MSHADOW_XINLINE static float Map(float x) { return sqrt(x); }
		};
		```
	5.  ``ms::expr::ReduceTo1DExp`` function for reduction to 1 dimension tensor, it can take as template parameter one of several predefined operations like ``minimum and maximum``, and as second parameter it takes scale factor (in out case 1), the first parameter is a tensor for reduction.
   
4. **Generating new data for testing model predictions**

  
5. **Batch gradient descent implementation**

   
6. **Generating additional polynomial components**

 
7. **Creating general regression model**

    
8. **Making predictions**

   
9. **Plot results**

    
You can find full source of this example on [GitHub](https://github.com/Kolkir/mlcpp).
<!--stackedit_data:
eyJoaXN0b3J5IjpbNzUwNjcwMjEyLDE0NzU5NDgyODIsMTY4Mj
cxNTY3MiwtMTIwODg4MjQwNywxOTczMzUyOTQ5LDI3Mjg1MzEx
MSwtMTQxNDczOTE1LDgxMjYxMjA5NCwxNzA3MjM2NjEzLC05Nj
k1NjU3MTAsNjgzMDEwODQsMTE3NzE4NjY2OSwxOTk5NzAyNzYy
LDE1Mjk2NDI2NDcsLTE3MzY0ODcyNDgsLTE3Mjk5NzY2NTddfQ
==
-->