This is C++ implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) with PyTorch C++ frontend.
The code is based on PyTorch implementations from [multimodallearning](https://github.com/multimodallearning/pytorch-mask-rcnn) and Keras implementation from [Matterport](https://github.com/matterport/Mask_RCNN). Also this implementation use heads from _resnet50_ and can be trained only with
[coco dataset](https://cocodataset.org/).

This implementation is compatible with PyTorch v1.0.1 .

**Development environment configuration**

I'm using Arch Linux, with additional packages ``openblas``, ``OpenCV``, ``gcc-7``, ``cuda``. At the moment when I was building PyTorch Cuda had support only for gcc-7 as host compiler, so you need to configure a build to use it. Don't use ``CC`` environmental variable for compiler configuration, because scripts depend on ``gcc``. To make build successful I used next strategy:
created a directory ``$HOME/old_gcc``, then symlink ``gcc`` in that directory to ``/usr/bin/gcc-7``. Add that directory to the front of PATH ``export PATH=$HOME/old_gcc:$PATH`` before building PyTorch.

Install PyYaml for your python environment ``pip install pyyaml``.

**PyTorch python wheel compilation**

_This step can be skipped if you don't need python environment with same version of PyTorch_

PyTorch has scripts for building library from sources, but before run them you need to setup environment variables, I used next ones:

* MAX_JOBS=8
* NO_FBGEMM=1
* NO_MKLDNN=1
* NO_NNPACK=1
* NO_QNNPACK=1
* ONNX_NAMESPACE=onnx_torch
* USE_OPENCV=1
* USE_OPENMP=1

And used next commands to build a wheel:
``` python
python setup.py bdist_wheel
pip install torch-1.0.0a0+4f0434d-cp37-cp37m-linux_x86_64.whl
```

**PyTorch C++ Frontend Compilation**

_If don't need a python wheel for PyTorch you can build only a C++ part. The previous step also builds the C++ frontend._

PyTorch has a CMake scripts, which can be used for build configuration and compilation. So you can use general procedure for building projects with CMake. I used next CMake command-line parameters to be able to build PyTorch in my environment:
* USE_CUDA=1
* USE_CUDNN=1
* USE_OPENCV=1
* USE_OPENMP=1
* BUILD_TORCH=1
* CMAKE_CXX_COMPILER=g++-7
* CMAKE_INSTALL_PREFIX="xxx"

I changed CMake parameter ``CMAKE_PREFIX_PATH`` to use custom directory for PyTorch installation.


**Application compilation**

1. After checking out of the code please update also submodules for the project, it have dependency for Eigen and Json parser libraries.
2. Update CMake parameter ``CMAKE_PREFIX_PATH`` with path where you installed PyTorch libraries, it will make ``find_package(Torch REQUIRED)`` works.

**Parameters management**

Please notice that parameters saved from python version of PyTorch with ``save_state_dict`` function are saved with pickle module, so are incompatible with C++ loading routings from PyTorch C++ frontend. How to manage parameters across language boundaries see code and comments in ``sateloader.h`` file.

**Using**

There are two projects ``mask-rcnn_demo`` and ``mask-rcnn_train`` which should be used with next parameters:
* *Demo* - ``mask-rcnn_demo`` executable takes two parameters ``path to file with trained parameters`` and ``path to image file for classification``. You can use pre-trained [parameters](https://drive.google.com/file/d/1H8_0uxCt7J7QIqQWs2QL-fW558-jRm9a/view?usp=sharing) from the original project (I just converted them to the format acceptable for C++ application). After processing you will get file, named ``result.png`` in your's working directory, with rendered bounding boxes, masks and printed labels. Command line can looks like this "mask-rcnn_demo checkpoint.pt test.png"

* *Train* - ``mask-rcnn_train`` executable takes twp parameters ``path to the coco dataset`` and ``path to the pretrained model``. If you want to start training from scratch, please put path to the pretrained resnet50 weights. Command line can looks like this "mask-rcnn_train /development/data/coco /development/model/resnet-50.pt". Default name for check-point file is ``./logs/checkpoint-epoch-NUM.pt``.

**Resources**
1. https://github.com/multimodallearning/pytorch-mask-rcnn
    * Branch with fixed  C++ extensions  https://github.com/mjstevens777/pytorch-mask-rcnn/tree/feat/build
2. https://github.com/matterport/Mask_RCNN
3. https://github.com/wannabeOG/Mask-RCNN
4. https://patrickwasp.com/create-your-own-coco-style-dataset/
