# Polynomial regression with MShadow library tutorial

Hello, this is my second article about how to use modern C++ for solving machine learning problems. This time I will show how to make a model for polynomial regression problem described in previous [article](https://github.com/Kolkir/mlcpp/tree/master/polynomial_regression), but now with another library which allows you to use your GPU easily.

For this tutorial I chose [MShadow](https://github.com/dmlc/mshadow) library, you can find documentation for it [here](https://github.com/dmlc/mshadow/tree/master/doc). This library was chosen because it is actively developed now, and used as a basis for one of a wide used deep learning framework [MXNet](https://mxnet.incubator.apache.org/). Also it is a header only library with minimal dependencies, so it's integration is not hard at all.

Please look at previous [article](https://github.com/Kolkir/mlcpp/tree/master/polynomial_regression) to learn how to download and parse training data. 

You have pay attention on how sources for this tutorial are compiled, I used CUDA compiler for them, please look at corresponding CMakeLists.txt file for details. Also you should have installed ``gcc-6`` as host compiler for ``CUDA 9``.

0. **Preparations**
	MShadow library use special routines to initialize and shutdown itself,  I wrote a simple class to use them in RAII manner:
	``` cpp
	#include <mshadow/tensor.h>
	template <typename Device>
	struct ScopedTensorEngine {
	  ScopedTensorEngine() { mshadow::InitTensorEngine<Device>(); }
	  ~ScopedTensorEngine() { mshadow::ShutdownTensorEngine<Device>(); }
	  ScopedTensorEngine(const ScopedTensorEngine&) = delete;
	  ScopedTensorEngine& operator=(const ScopedTensorEngine&) = delete;
	};
	```
	Device template parameter can be ``mshadow::cpu`` or ``mshadow::gpu``, I will instantiate ``ScopedTensorEngine`` for both types, because I need to pass data from host side to GPU (but it is allowed to use only ``mshadow::cpu`` for all computations, and code will remain the same):
	```cpp
	ScopedTensorEngine<mshadow::cpu> tensorEngineCpu;
	ScopedTensorEngine<mshadow::gpu> tensorEngineGpu;
	```
	Take a look on ``USE_GPU`` define in source code, it allows you to disable using GPU and run example only on CPU.
	```cpp
	#ifdef USE_GPU  // use macros because lack of "if constexpr" in c++14
	using xpu = mshadow::gpu;
	#else
	using xpu = mshadow::cpu;
	#endif
	```
	Next I defined a variable which will represent a CUDA stream. A CUDA Stream is a sequence of operations that are performed in order on the GPU device. Streams can be run in independent concurrent in-order queues of execution, and operations in different streams can be interleaved and overlapped. This variable is necessary for using other MShadow abstractions. 
	```cpp
	using DType = float;
	using GpuStream = mshadow::Stream<xpu>;
	using GpuStreamPtr = std::unique_ptr<GpuStream, void (*)(GpuStream*)>;
	```
	C++ smart pointer with custom deleter can be very useful for C style interfaces.  
	
1. **Loading data to MShadow datastructures**
	There are several approaches to initialize tensors data structures in MShadow library, two of them I used next code section.
	```cpp
	template <typename Device, typename DType>
	void load_data(std::vector<DType>& raw_data,
	               mshadow::TensorContainer<Device, 2, DType>& dst) {
	  mshadow::Tensor<mshadow::cpu, 2, DType> host_data(
	      raw_data.data(), mshadow::Shape2(raw_data.size(), 1));
	  dst.Resize(host_data.shape_);
	  mshadow::Copy(dst, host_data, dst.stream_);
	}
	...
	mshadow::TensorContainer<xpu, 2, DType> x;
	x.set_stream(computeStream.get());
	load_data<xpu>(raw_data_x, x);
	
	mshadow::TensorContainer<xpu, 2, DType> y;
    y.set_stream(computeStream.get());
    load_data<xpu>(raw_data_y, y);
	```
    When I initialize ``host_data`` variable I provide pointer to raw data array in constructor, so in this case tensor will work as wrapper around raw array. It's very useful technique to work with host data to eliminate unnecessary copying.  Next I used ``mshadow::TensorContainer`` type which implements RAII idiom for ``mshadow::Tensor``, it will allocate required amount of memory and free it in a destructor.  I found it useful for managing GPU data, but library authors recommend it mostly for intermediate calculations results. Also pay attention on how CUDA stream is used, for ``x`` initialization and during copy operation. 
    
2. **Standardization**
To be able to perform successful computations for regression analysis we need to [standardize](https://en.wikipedia.org/wiki/Feature_scaling#Standardization) our data. Also because we need to pre-allocate several  intermediate tensors for calculations and to reuse a code I implemented standardization procedure as separate class.
	```cpp
	// Standardize 2D tensor of shape [rows]x[1]
	template <typename Device, typename DType>
	class Standardizer {
	 public:
	  using Tensor = mshadow::TensorContainer<Device, 2, DType>;
	  using Stream = mshadow::Stream<Device>;
	  
	  Standardizer() {}
	  ~Standardizer() {}
	  Standardizer(const Standardizer&) = delete;
	  Standardizer& operator=(const Standardizer&) = delete;

	  void transform(Tensor& vec) {
	    assert(vec.shape_.kDimension == 2);
	    assert(vec.shape_[1] == 1);

	    auto rows = vec.shape_[0];

	    // alloc dst/temp tensors
	    mean.Resize(mshadow::Shape1(1));
	    mean.set_stream(vec.stream_);
	    temp.Resize(mshadow::Shape2(rows, 1));
	    temp.set_stream(vec.stream_);
	    sd.Resize(mshadow::Shape1(1));
	    sd.set_stream(vec.stream_);

	    // calculate
	    mean = mshadow::expr::sumall_except_dim<1>(vec);
	    mean /= static_cast<DType>(rows);
	    temp = mshadow::expr::F<Pow>(
	        vec - mshadow::expr::broadcast<1>(mean, temp.shape_), 2);

	    sd = mshadow::expr::sumall_except_dim<1>(temp);
	    sd = mshadow::expr::F<Sqrt>(sd / static_cast<DType>(rows - 1));

	    temp = (vec - mshadow::expr::broadcast<1>(mean, temp.shape_)) /
	           mshadow::expr::broadcast<1>(sd, temp.shape_);

	    mshadow::Copy(vec, temp, vec.stream_);
	  }

	  auto get_moments() {
	    mshadow::TensorContainer<mshadow::cpu, 1, DType> value(mshadow::Shape1(1));
	    mshadow::Copy(value, mean, mean.stream_);
	    DType v_mean = value[0];
	    mshadow::Copy(value, sd, sd.stream_);
	    DType v_sd = value[0];
	    return std::vector<DType>{v_mean, v_sd};
	  }

	 private:
	  mshadow::TensorContainer<Device, 1, DType> mean;
	  mshadow::TensorContainer<Device, 1, DType> sd;
	  mshadow::TensorContainer<Device, 2, DType> temp;
	};
	...
	...
	// standardize data
	auto rows = raw_data_x.size();
	Standardizer<xpu, DType> standardizer;
	standardizer.transform(x);
	standardizer.transform(y);
	auto y_moments = standardizer.get_moments(); // used later for scale restoring
	``` 
	The interesting moments here are :
	1.  ``mshadow::expr::broadcast`` function which make possible to define element wise operations for tensors with single value, for example subtraction one number from each tensor element. There is a dynamic broadcasting in this library, but to use it you need actual value (it doesn't work for expressions), so in some cases it requires earlier expression evaluation which can hurt performance.
	2.  ``mshadow::expr::sumall_except_dim`` function which calculate sum of elements along not specified tensor dimension. 
	3.  ``mshadow::expr::F`` custom user specified operation on tensor elements, I used power and square root operations:
		```cpp
		struct Pow {
		  MSHADOW_XINLINE static float Map(float x, float y) { return pow(x, y); }
		};
		
		struct Sqrt {
		  MSHADOW_XINLINE static float Map(float x) { return sqrt(x); }
		};
		```
   
3. **Generating additional polynomial components**
Before 

4. **Generating new data for testing model predictions**

5. **Batch gradient descent implementation**
 
6. **Creating general regression model**
    
7. **Making predictions**
   
8. **Plot results**

    
You can find full source of this example on [GitHub](https://github.com/Kolkir/mlcpp).
<!--stackedit_data:
eyJoaXN0b3J5IjpbNDMyODkyNjI4LDE1MjQxNjAxMjAsMTkxOD
E5NjQ3NSw1Mjk5ODI0ODksLTE0NDg2NTEzMyw1MDA5OTk2MDgs
LTE3MTM0MTc4MCwxNTQ1ODU4NDg3LC0xNjU5NDI5MjMsNzUwNj
cwMjEyLDE0NzU5NDgyODIsMTY4MjcxNTY3MiwtMTIwODg4MjQw
NywxOTczMzUyOTQ5LDI3Mjg1MzExMSwtMTQxNDczOTE1LDgxMj
YxMjA5NCwxNzA3MjM2NjEzLC05Njk1NjU3MTAsNjgzMDEwODRd
fQ==
-->