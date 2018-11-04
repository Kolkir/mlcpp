This is an implementation of Faster R-CNN with MXNet C++ frontend. 
The code is based on python implementation from [ijkguo](https://github.com/ijkguo/mx-rcnn).

To be able to compile the code you should build MXNet from source code, 
with enabled Cpp Package, you can use the original [tutorial](https://github.com/apache/incubator-mxnet/tree/master/cpp-package) for this.

Because I'm using Arch Linux, I'm sharing next tips which helped me to
built MXNet for this platform:

1. Install additional packages ``openblas``, ``OpenCV``, ``gcc-7``
2. I used CMake to configure MXNet build
3. Used ``CMAKE_CXX_COMPILER=g++-7`` CMake parameter. (It resolved problems with tbb version used in MXNet)
4. Used ``OLDCMAKECUDA`` CMake parameter, to be able to use Cuda >= 9.
5. Used ``USE_CPP_PACKAGE`` CMake parameter, to enable MXNet Cpp Package (C++ frontend).
6. Used ``USE_OPENCV`` CMake parameter, to integrate MXNet with installed version of OpenCV.

To Use compiled MXNet library some additional steps are required: 

1. MXNet build script downloads mkldnn_intel library, but install script don't copy
it to installation folder, so copy it manually and specify "mkldtnn" for linker.
2. MXNet Cpp Package header files are not copied to install folder automatically - so do it manually,
provide path to ``mxnet-cpp`` folder as include source for a compiler.
3. ``nnvm`` header files are also missed in the install folder, so manually specify path to them for a compiler.
4. It have sense to suppress -Wunused-parameter warnings, because they will foul compiler output.

If you are used not the top GPU you can reach a problem with cuda error ``too many resources requested for launch`` 
it can be solved by adding ``MSHADOW_CFLAGS += -DMSHADOW_OLD_CUDA=1`` to ``mxnet/mshadow/make/mshadow.mk`` file.
This macro limits the threads of kernel launch, refer this 
[link](https://github.com/dmlc/mshadow/blob/3a400b4662bf42e885592fd07bd51335532a8bc8/mshadow/cuda/tensor_gpu-inl.cuh#L25) for more details.
After making such change you will need to re-compile MXNet.
