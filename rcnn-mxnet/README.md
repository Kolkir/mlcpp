This is an implementation of Faster R-CNN with MXNet C++ frontend. 
The code is based on python implementation from [ijkguo](https://github.com/ijkguo/mx-rcnn), 
but implements model only with ``resnet 101`` head and can be trained only with 
[coco dataset](https://cocodataset.org/).

**MXNet Compilation**

To be able to compile the code you should build MXNet from source code, 
with enabled Cpp Package, you can use the original [tutorial](https://github.com/apache/incubator-mxnet/tree/master/cpp-package) for this.

Because I'm using Arch Linux, I'm sharing next tips which helped me to
built MXNet for this platform:

1. Install additional packages ``openblas``, ``OpenCV``, ``gcc-7``, ``cuda``
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

**Application compilation**

1. After checking out of the code please update also submodules for the project, it have dependency for Eigen and Json parser libraries.
2. You should use CMake to configure build scripts for compilation, please provide value for ``LIBRARIES_DIR`` CMake parameter with a path to the directory where MXNet is installed in your environment. 

**Using**

There are two projects ``demo`` and ``train`` which should be used with next parameters:
* *Demo* - ``rcnn_demo`` executable takes two parameters ``path to file with trained parameters`` and ``path to image file for classification``. You can use pre-trained [parameters](https://www.dropbox.com/s/bfuy2uo1q1nwqjr/resnet_coco-0010.params?dl=0) from the original project. After processing you will get file, named ``det.png`` in your's working directory, with rendered bounding boxes and printed labels. Also application will print classification results to the standard output. Commandline can looks like this "rcnn_demo check-point.params test.png"

* *Train* - ``rcnn_train`` executable takes next parameters ``path to the coco dataset``, ``path to the pretrained resnet model``, flag ``--start-train`` which means starting training from scratch or ``path to the file with saved check-point paramenters``. Commandline can looks like this "rcnn_train /development/data/coco --params=/development/model/resnet-101-0000.params --start-train". Default name for check-point file is ``check-point.params``. You can download pre-trained resnet parameters from [MXNet model zoo](http://data.dmlc.ml/models/imagenet/resnet/101-layers/). 

Also you can download file with pre-trained parameters from this [link](https://drive.google.com/file/d/1WMC9TvawKrz7Jjc4V8O5pryuyaR2z96y/view?usp=sharing), it was made for proof of the concept and for vehicles label types only also it was trained on small number of iteration, because I don't have suitable hardware for full training cycle.

**Notes**

Please look in the source code for details of implementation, I left comments in the most interesting parts. This implementation has custom ``proposal target`` layer as part of the project and don't required MXNet library modification. Also it has custom loader for Coco dataset because I did not find existent one for C++. One of the biggest problem during development was absence of normal error reporting from MXNet C++ frontend (Usually API simply ignore all errors reported from C API), so I added some wrappers around C API to be aware of errors during runtime. Another tricky part was synchronising memory layout of Eigen data structures with MXNet NDArray class, please pay attention on this in the code if you will modify it.
