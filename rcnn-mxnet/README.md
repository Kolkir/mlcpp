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

