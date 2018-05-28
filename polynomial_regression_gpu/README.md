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
	Before generating actual polynomial components, we need to scale our data to an appropriate range before raise to power to prevent float overflow in the optimizer, this is restriction of ``float`` type.  A scale factor was chosen after several experiments with polynomial degree of 64.
	```cpp
	DType scale = 0.6;
	x *= scale;
	y *= scale;
	```
	Here you can see the example of a dynamic broadcasting. To make additional polynomial components I just raise to power from ``1`` no ``n`` each sample from ``X``:
	``` cpp
	template <typename Device, typename DType>
	void generate_polynomial(mshadow::Tensor<Device, 2, DType> const& tensor,
	                         mshadow::TensorContainer<Device, 2, DType>& poly,
	                         size_t p_degree) {
	  assert(tensor.shape_.kDimension == 2);
	  assert(tensor.shape_[1] == 1);

	  auto rows = tensor.shape_[0];
	  mshadow::TensorContainer<Device, 2, DType> col_temp(mshadow::Shape2(rows, 1));
	  col_temp.set_stream(tensor.stream_);

	  for (size_t c = 0; c < p_degree; ++c) {
	    auto col = mshadow::expr::slice(poly, mshadow::Shape2(0, c),
	                                    mshadow::Shape2(rows, c + 1));
	    col_temp = mshadow::expr::F<Pow>(tensor, static_cast<DType>(c));
	    col = col_temp;
	  }
	}
	...
	size_t p_degree = 64;
	mshadow::TensorContainer<xpu, 2, DType> poly_x(mshadow::Shape2(rows, p_degree));
	poly_x.set_stream(computeStream.get());
	generate_polynomial(x, poly_x, p_degree);
	``` 
	The most interesting thing here is function ``mshadow::expr::slice`` which produce a references slice from original tensor and you can use it as separate tensor object in expressions. I didn't make function ``generate_polinomial``  return a ``TensorContainer`` object, because there is a missing of explicit ``Tensor`` object initialization in its copy constructor which leads to compiler warnings.
	 
4. **Generating new data for testing model predictions**
	Generating new data is very straight forward, I generate contiguous values from min value to max value of original ``X``, with constant step which is defined by total number of values.  The new data are also standardized and scaled, and additional polynomial components are generated.
	``` cpp
	  size_t n = 2000;
	  auto minmax_x = std::minmax_element(raw_data_x.begin(), raw_data_x.end());
	  auto time_range = *minmax_x.second - *minmax_x.first;
	  auto inc_step = time_range / n;
	  auto x_val = inc_step;
	  std::vector<DType> new_data_x(n);
	  for (auto& x : new_data_x) {
	    x = x_val;
	    x_val += inc_step;
	  }
	  mshadow::TensorContainer<xpu, 2, DType> new_x(mshadow::Shape2(n, 1));
	  new_x.set_stream(computeStream.get());
	  load_data<xpu>(new_data_x, new_x);
	  standardizer.transform(new_x);
	  new_x *= scale;

	  mshadow::TensorContainer<xpu, 2, DType> new_poly_x(
	      mshadow::Shape2(n, p_degree));
	  new_poly_x.set_stream(computeStream.get());
	  generate_polynomial(new_x, new_poly_x, p_degree);
	```

6. **Batch gradient descent implementation**
	 For this example a code for learning model and results predicting I moved to separate class. It helps to reuse code more easily and make its usage more clear. Also here I implemented  [AdaDelta](https://arxiv.org/abs/1212.5701) optimizing technique, because it make learning process to converge quicker and  dynamically adapts learning rate. You should pay attention on next things:  resizing all tensors before actual usage, using ``mshadow::expr::dot`` function for tensors(matrix) multiplication, using ``Slice`` function for batchs extracting nad using 
	``` cpp
	template <typename Device, typename DType>
	class Optimizer {
	 public:
	  Optimizer() {}

	  void predict(mshadow::Tensor<Device, 2, DType> const& x,
	               mshadow::Tensor<Device, 2, DType>& y) {
	    y = mshadow::expr::dot(x, weights);
	  }

	  void fit(mshadow::Tensor<Device, 2, DType> const& x,
	           mshadow::Tensor<Device, 2, DType> const& y) {
	    assert(y.shape_.kDimension == 2);
	    assert(y.shape_[1] == 1);
	    assert(x.shape_.kDimension == 2);
	    assert(x.shape_[0] == y.shape_[0]);

	    size_t cols = x.shape_[1];
	    size_t rows = x.shape_[0];
	    size_t n_batches = rows / batch_size;

	    // it is important to allocate all tensors before assiging
	    weights.set_stream(x.stream_);
	    weights.Resize(mshadow::Shape2(cols, 1));
	    weights = 0.0f;

	    grad.Resize(mshadow::Shape2(cols, 1));
	    grad.set_stream(x.stream_);

	    yhat.Resize(mshadow::Shape2(batch_size, 1));
	    yhat.set_stream(x.stream_);

	    error.Resize(mshadow::Shape2(batch_size, 1));
	    error.set_stream(x.stream_);

	    error_total.Resize(mshadow::Shape2(rows, 1));
	    error_total.set_stream(x.stream_);

	    error_total_cpu.Resize(mshadow::Shape2(rows, 1));

	    eg_sum.Resize(mshadow::Shape2(cols, 1));
	    eg_sum.set_stream(x.stream_);
	    eg_sum = 0.f;

	    weights_delta.Resize(mshadow::Shape2(cols, 1));
	    weights_delta.set_stream(x.stream_);

	    ex_sum.Resize(mshadow::Shape2(cols, 1));
	    ex_sum.set_stream(x.stream_);
	    ex_sum = 0.f;

	    // gradient descent
	    for (size_t epoch = 0; epoch < n_epochs; ++epoch) {
	      for (size_t bi = 0; bi < n_batches; ++bi) {
	        auto bs = bi * batch_size;
	        auto be = bs + batch_size;
	        auto batch_x = x.Slice(bs, be);
	        auto batch_y = y.Slice(bs, be);
	        
	        predict(batch_x, yhat);

	        error = yhat - batch_y;
	        grad = mshadow::expr::dot(batch_x.T(), error);
	        grad /= batch_size;

	        // AdaDelta
	        eg_sum = lr * eg_sum + (1.f - lr) * mshadow::expr::F<Pow>(grad, 2);
	        weights_delta = -1.f *
	                        (mshadow::expr::F<Sqrt>(ex_sum + e) /
	                         mshadow::expr::F<Sqrt>(eg_sum + e)) *
	                        grad;
	        ex_sum =
	            lr * ex_sum + (1.f - lr) * mshadow::expr::F<Pow>(weights_delta, 2);
	        weights = weights + weights_delta;
	      }
	      // compute cost
	      error_total = mshadow::expr::dot(x, weights);
	      error_total = mshadow::expr::F<Pow>(error_total - y, 2);

	      mshadow::Copy(error_total_cpu, error_total, error_total.stream_);
	      long double cost = 0;
	      for (size_t r = 0; r < rows; ++r)
	        cost += error_total_cpu[r][0];
	      cost /= rows;
	      std::cout << "Epoch " << epoch << " cost = " << cost << std::endl;
	    }
	  }

	 private:
	  size_t n_epochs = 5000;
	  size_t batch_size = 8;

	  mshadow::TensorContainer<Device, 2, DType> weights;
	  mshadow::TensorContainer<Device, 2, DType> grad;
	  mshadow::TensorContainer<Device, 2, DType> yhat;
	  mshadow::TensorContainer<Device, 2, DType> error;
	  mshadow::TensorContainer<Device, 2, DType> error_total;
	  mshadow::TensorContainer<mshadow::cpu, 2, DType> error_total_cpu;

	  // AdaDelta
	  DType lr = 0.99;
	  DType e = std::numeric_limits<DType>::epsilon();
	  mshadow::TensorContainer<Device, 2, DType> eg_sum;
	  mshadow::TensorContainer<Device, 2, DType> weights_delta;
	  mshadow::TensorContainer<Device, 2, DType> ex_sum;
	};
	``` 
 
7. **Training the regression model**
    
8. **Making predictions**
   
9. **Plot results**

    
You can find full source of this example on [GitHub](https://github.com/Kolkir/mlcpp).
<!--stackedit_data:
eyJoaXN0b3J5IjpbOTQ4MTAxODUxLDk5MTE5OTYyNiwtMTk4MD
I5MTA5OSwyMTM5MjE5MTc5LDk4MzQxMzY4OCw3ODY3Njc5ODcs
NzIwMzc5NjEsLTU5MDY0NTI2MCw0ODA3NTY5OTYsMTE3NzEyNz
c4LC0zNDc1MjMxNzIsMTUyNDE2MDEyMCwxOTE4MTk2NDc1LDUy
OTk4MjQ4OSwtMTQ0ODY1MTMzLDUwMDk5OTYwOCwtMTcxMzQxNz
gwLDE1NDU4NTg0ODcsLTE2NTk0MjkyMyw3NTA2NzAyMTJdfQ==

-->