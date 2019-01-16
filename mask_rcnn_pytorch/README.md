**Resources**
1. https://github.com/multimodallearning/pytorch-mask-rcnn
    * Fixed  C++ extensions  https://github.com/mjstevens777/pytorch-mask-rcnn/tree/feat/build
        * change order of arguments like this one - at::zeros({col_blocks}, at::CPU(at::kLong));
2. https://github.com/matterport/Mask_RCNN
3. https://github.com/wannabeOG/Mask-RCNN

**PyTorch python wheel compilation**
* CC=gcc-7
* CXX=g++-7
* MAX_JOBS=8
* NO_FBGEMM=1 ?
* NO_MKLDNN=1
* NO_NNPACK=1
* NO_QNNPACK=1
* ONNX_NAMESPACE=onnx_torch
* USE_OPENCV=1
* USE_OPENMP=1

Wheel building process don't use ``CC`` variable and depends on ``gcc`` so:
Create a directory ``$HOME/old_gcc``, symlink ``gcc`` in that directory to ``/usr/bin/gcc-7``. Add that directory to the front of PATH before building software that requires the old gcc: ``export PATH=$HOME/old_gcc:$PATH``.

Use next commands to build wheel:
``` python
python setup.py bdist_wheel
pip install torch-1.0.0a0+4f0434d-cp37-cp37m-linux_x86_64.whl 
```


**PyTorch C++ Frontend compilation**
1. Checkout
2. ``pip install pyyaml``
3. Run ``CMake``, enable:
     * USE_CUDA,
     * USE_CUDNN,
     * USE_OPENCV,
     * USE_OPENMP,
     * BUILD_TORCH
     * CMAKE_INSTALL_PREFIX
     * CMAKE_CXX_COMPILER=g++-7
4. For project specify ``CMAKE_PREFIX_PATH=directory where you install Torch``, if it is not a default system folders

**Sharing weights**
torch.save/load - use python pickle.
torch.jit - can be used with tracing so in case complex training function it is tricky to use.

**Access data
auto t = *y.data<uint8_t>();  // see at::ScalarType::Byte use data for scalars
also [] operator can be used

**Create tensors
https://pytorch.org/cppdocs/notes/tensor_creation.html
see torch::tensor function for creation tensor from initializer list

**Torch cpu have unstable reductions(max, argmax) that differs from numpy. Try gpu
