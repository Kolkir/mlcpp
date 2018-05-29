# Polynomial regression with MShadow library tutorial

Hello, this is my second article about how to use modern C++ for solving machine learning problems. This time I will show how to make a model for polynomial regression problem described in previous [article](https://github.com/Kolkir/mlcpp/tree/master/polynomial_regression), but now with another library which allows you to use your GPU easily.

For this tutorial I chose [MShadow](https://github.com/dmlc/mshadow) library, you can find documentation for it [here](https://github.com/dmlc/mshadow/tree/master/doc). This library was chosen because it is actively developed now, and used as a basis for one of a wide used deep learning framework [MXNet](https://mxnet.incubator.apache.org/). Also it is a header only library with minimal dependencies, so it's integration is not hard at all.

Please look at previous [article](https://github.com/Kolkir/mlcpp/tree/master/polynomial_regression) to learn how to download and parse training data.

You have pay attention on how sources for this tutorial are compiled, I used CUDA compiler for them, please look at corresponding CMakeLists.txt file for details. Also you should have installed ``gcc-6`` as host compiler for ``CUDA 9``.

0. **Preparations**

	MShadow library use special routines to initialize and shutdown itself,  I wrote a simple class to use them in RAII manner:
	```cpp
	#include <mshadow/tensor.h>
	template <typename Device>
	struct ScopedTensorEngine {
	  ScopedTensorEngine() { mshadow::InitTensorEngine<Device>(); }
	  ~ScopedTensorEngine() { mshadow::ShutdownTensorEngine<Device>(); }
	  ScopedTensorEngine(const ScopedTensorEngine&) = delete;
	  ScopedTensorEngine& operator=(const ScopedTensorEngine&) = delete;
	};
	```
	Device template parameter can be ``mshadow::cpu`` or ``mshadow::gpu``, I instantiated ``ScopedTensorEngine`` for both types, because I need to pass data from host side to GPU (but it is allowed to use only ``mshadow::cpu`` for all computations, and code will remain the same):
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

1. **Loading data to MShadow data-structures**

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
	When I initialized ``host_data`` variable I provided a pointer to raw data array in constructor, so in this case tensor will work as wrapper around raw array. It's very useful technique to work with host data to eliminate unnecessary copying.  Next I used ``mshadow::TensorContainer`` type which implements RAII idiom for ``mshadow::Tensor``, it allocates required amount of memory and free it in a destructor.  I found it useful for managing GPU data, but library authors recommend to use it mostly for intermediate calculations results. Also pay attention on how CUDA stream is used, for ``x`` initialization and during copy operation. Also you can define a variable of type ``TensorContainer`` and resize it later:
	```cpp
	mshadow::TensorContainer<Device, 1, DType> mean;
	...
	mean.Resize(mshadow::Shape1(1));
	mean.set_stream(vec.stream_);
	```
	I used such technique for lazy initialization of class members.

2. **Standardization**

	To be able to perform successful computations for regression analysis we need to [standardize](https://en.wikipedia.org/wiki/Feature_scaling#Standardization) our data. Also because we need to pre-allocate several  intermediate tensors for calculations and to reuse a code I implemented standardization procedure as separate class.
	```cpp
	// Standardize 2D tensor of shape [rows]x[1]
	template <typename Device, typename DType>
	class Standardizer {
	 public:
	  using Tensor = mshadow::TensorContainer<Device, 2, DType>;
	  ...
	  void transform(Tensor& vec) {
	    ...
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
	  ...
	};
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
	  MSHADOW_XINLINE static float Map(float x, float y) { return pow(x, y);}
	};

	struct Sqrt {
	  MSHADOW_XINLINE static float Map(float x) { return sqrt(x); }
	};
	```

3. **Generating additional polynomial components**

	Before generating actual polynomial components, we need to scale our data to an appropriate range before raising to a power to prevent ``float`` type overflow in the optimizer.  A scale factor was chosen after several experiments with polynomial degree of 64.
	```cpp
	DType scale = 0.6;
	x *= scale;
	y *= scale;
	```
	Here you can see the example of a dynamic broadcasting. To make additional polynomial components I just raise to power from ``1`` no ``n`` each sample from ``X`` data set (where ``n`` is a polynomial degree):
	```cpp
	template <typename Device, typename DType>
	void generate_polynomial(mshadow::Tensor<Device, 2, DType> const& tensor,
	                         mshadow::TensorContainer<Device, 2, DType>& poly,
	                         size_t p_degree) {
	  ...
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
	The most interesting thing here is function ``mshadow::expr::slice`` which produce a references slice from original tensor and you can use it as separate tensor object in expressions. I didn't make function ``generate_polinomial`` return a ``TensorContainer`` object, because there is a missing of explicit ``Tensor`` object initialization in its copy constructor which leads to compiler warnings.

4. **Generating new data for testing model predictions**

	Generating new data is very straight forward, I generate contiguous values from ``min`` value to ``max`` value of original ``X`` data set, with constant step, which is defined by total number of values.  The new data are also standardized and scaled, and additional polynomial components are generated.
	```cpp
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

5. **Batch gradient descent implementation**

	For this example code for learning model and results predicting I moved to separate class. It helps to reuse code more easily and make its usage more clear. Also here I implemented  [AdaDelta](https://arxiv.org/abs/1212.5701) optimizing technique, because it makes learning process to converge quicker and  dynamically adapts learning rate during training. You should pay attention on next things:
		1. using ``mshadow::expr::dot`` function for tensors(matrix) multiplication
		2. using ``Slice`` function for batches extracting
		3. usung ``T()`` method of tensor object for taking a transposed one.

	```cpp
	template <typename Device, typename DType>
	class Optimizer {
	 public:

	  void predict(mshadow::Tensor<Device, 2, DType> const& x,
	               mshadow::Tensor<Device, 2, DType>& y) {
	    y = mshadow::expr::dot(x, weights);
	  }

	  void fit(mshadow::Tensor<Device, 2, DType> const& x,
	           mshadow::Tensor<Device, 2, DType> const& y) {
	    size_t cols = x.shape_[1];
	    size_t rows = x.shape_[0];
	    size_t n_batches = rows / batch_size;
	    ...
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
	      ...
	    }
	  }
	...
	};
	```
	``predict`` method doesn't return a value and takes an output parameter to prevent compiler warnings.

6. **Training the regression model**

	With class defined above I can run training pretty easily:
	```cpp
	Optimizer<xpu, DType> optimizer;
	optimizer.fit(poly_x, y);
	```

7. **Making predictions**

	Predictions also are straight forward:
	```cpp
	mshadow::TensorContainer<xpu, 2, DType> new_y(mshadow::Shape2(n, 1));
	new_y.set_stream(computeStream.get());
	optimizer.predict(new_poly_x, new_y);
	```
	But before actual using of predicted values we need to restore scaling and undo standardization (our model learned to return such types of values):
	```cpp
	new_y /= scale;
	new_y = (new_y * y_moments[1]) + y_moments[0];
	```
	Here ``y_moments[1]`` is a standard deviation and ``y_moments[0]`` is a mean.

9. **Plot results**

	To plot results I moved  predicted values to C++ vector data structure to have iterators compatible with a plotting library:
	```cpp
	std::vector<DType> raw_pred_y(n);
	mshadow::Tensor<mshadow::cpu, 2, DType> pred_y(raw_pred_y.data(),mshadow::Shape2(n, 1));
	mshadow::Copy(pred_y, new_y, computeStream.get());
	...
	plotcpp::Plot plt(true);
	...
	plt.Draw2D(
    plotcpp::Points(raw_data_x.begin(), raw_data_x.end(), raw_data_y.begin(),
                      "points", "lc rgb 'black' pt 1"),
    plotcpp::Lines(new_data_x.begin(), new_data_x.end(), raw_pred_y.begin(),
                     "poly line approx", "lc rgb 'green' lw 2"));
    plt.Flush();
	```
	![plots](plot.png)


You can find full source of this example on [GitHub](https://github.com/Kolkir/mlcpp).
