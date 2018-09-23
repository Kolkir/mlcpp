**Install MXNet**

Install openblas
I used CMake to configure MXNet build
Set CMAKE_CXX_COMPILER=g++-7 - requied for tbb version MXNet use
Set OLDCMAKECUDA - You use Cuda >= 9
Set USE_CPP_PACKAGE
Set USE_OPENCV

MXNet build script downloads mkldnn_intel library, but install script don't copy
it to specified place, copy it manually and specify "mkldnn" for linker

cpp-package includes are not copiest to install folder automatically - do it manually
provide path to mxnet-cpp folder as include source for a compiler

nnvm - include path missed in install folder

Alot of -Wunused-parameter warnings

Solve the cuda "too many resources requested for launch" problem by adding MSHADOW_CFLAGS += -DMSHADOW_OLD_CUDA=1 in mxnet/mshadow/make/mshadow.mk.
This macro limits the threads of kernel launch, refer https://github.com/dmlc/mshadow/blob/3a400b4662bf42e885592fd07bd51335532a8bc8/mshadow/cuda/tensor_gpu-inl.cuh#L25 for more details.
And re-compile mxnet.
