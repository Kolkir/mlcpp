#ifndef CROPANDRESIZE_KERNEL
#define CROPANDRESIZE_KERNEL

#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

void CropAndResizeLaucher(const float* image_ptr,
                          const float* boxes_ptr,
                          const int64_t* box_ind_ptr,
                          int64_t num_boxes,
                          int64_t batch,
                          int64_t image_height,
                          int64_t image_width,
                          int64_t crop_height,
                          int64_t crop_width,
                          int64_t depth,
                          float extrapolation_value,
                          float* crops_ptr);

void CropAndResizeBackpropImageLaucher(const float* grads_ptr,
                                       const float* boxes_ptr,
                                       const int64_t* box_ind_ptr,
                                       int64_t num_boxes,
                                       int64_t batch,
                                       int64_t image_height,
                                       int64_t image_width,
                                       int64_t crop_height,
                                       int64_t crop_width,
                                       int64_t depth,
                                       float* grads_image_ptr);

#ifdef __cplusplus
}
#endif

#endif
