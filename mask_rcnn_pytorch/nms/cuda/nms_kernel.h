#ifndef _NMS_KERNEL
#define _NMS_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(int64_t) * 8;

void _nms(int boxes_num, float * boxes_dev,
          int64_t * mask_dev, float nms_overlap_thresh);

#ifdef __cplusplus
}
#endif

#endif

