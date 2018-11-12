**Resources**
1. https://github.com/multimodallearning/pytorch-mask-rcnn
2. https://github.com/matterport/Mask_RCNN
3. https://github.com/wannabeOG/Mask-RCNN

**PyTorch C++ Frontend compilation**
1. Checkout
2. pip install pyyaml
3. Run CMake, enable:
     USE_CUDA,
     USE_CUDNN,
     USE_OPENCV,
     USE_OPENMP,
     BUILD_TORCH
     CMAKE_INSTALL_PREFIX
     CMAKE_CXX_COMPILER=g++-7
4. For project specify CMAKE_PREFIX_PATH=directory where you install Torch, if it is not a default system folders
