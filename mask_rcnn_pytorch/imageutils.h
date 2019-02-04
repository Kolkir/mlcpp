#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H

#include "config.h"

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

cv::Mat LoadImage(const std::string path);

at::Tensor CvImageToTensor(const cv::Mat& image);

struct Window {
  int32_t y1{0};
  int32_t x1{0};
  int32_t y2{0};
  int32_t x2{0};
};

struct Padding {
  int32_t top_pad{0};
  int32_t bottom_pad{0};
  int32_t left_pad{0};
  int32_t right_pad{0};
  int32_t front_pad{0};
  int32_t rear_pad{0};
};

struct ImageMeta {
  int32_t image_id{0};
  int32_t image_width{0};
  int32_t image_height{0};
  Window window;
  // std::vector<int32_t> active_class_ids; Unused
};

/*
 * Resizes an image keeping the aspect ratio.
 *
 *  min_dim: if provided, resizes the image such that it's smaller
 *      dimension == min_dim
 *  max_dim: if provided, ensures that the image longest side doesn't
 *      exceed this value.
 *  padding: If true, pads image with zeros so it's size is max_dim x max_dim
 *
 *  Returns:
 *  image: the resized image
 *  window: (y1, x1, y2, x2). If max_dim is provided, padding might
 *      be inserted in the returned image. If so, this window is the
 *      coordinates of the image part of the full image (excluding
 *      the padding). The x2, y2 pixels are not included.
 *  scale: The scale factor used to resize the image
 *  padding: Padding added to the image
 */
std::tuple<cv::Mat, Window, float, Padding> ResizeImage(
    cv::Mat image,
    int32_t min_dim,
    int32_t max_dim,
    bool do_padding = false);

/* Resizes a mask using the given scale and padding.
 * Typically, you get the scale and padding from resize_image() to
 * ensure both, the image and the mask, are resized consistently.
 * scale: mask scaling factor
 * padding: Padding to add to the mask
 */
std::vector<cv::Mat> ResizeMasks(const std::vector<cv::Mat>& masks,
                                 float scale,
                                 const Padding& padding);

/* Resize masks to a smaller version to cut memory load.
 */
std::vector<cv::Mat> MinimizeMasks(const std::vector<float>& boxes,
                                   const std::vector<cv::Mat>& masks,
                                   int32_t width,
                                   int32_t height);

/*
 * Takes RGB images with 0-255 values and subtraces
 * the mean pixel and converts it to float. Expects image
 * colors in RGB order.
 */
cv::Mat MoldImage(cv::Mat image, const Config& config);

/*
 * Takes a list of images and modifies them to the format expected
 * as an input to the neural network.
 * images: List of image matricies [height,width,depth]. Images can
 * have different sizes.
 * Returns 3 matricies: molded_images: [N, h, w, 3].
 * Images resized and normalized. image_metas: [N, length of meta data]. Details
 * about each image. windows: [N, (y1, x1, y2, x2)]. The portion of the image
 * that has the original image (padding excluded).
 */
std::tuple<at::Tensor, std::vector<ImageMeta>, std::vector<Window>> MoldInputs(
    const std::vector<cv::Mat>& images,
    const Config& config);

/*
 * Reformats the detections of one image from the format of the neural
 * network output to a format suitable for use in the rest of the
 * application.
 * detections: [N, (y1, x1, y2, x2, class_id, score)]
 * mrcnn_mask: [N, height, width, num_classes]
 * image_shape: [height, width, depth] Original size of the image before
 resizing
 * window: [y1, x1, y2, x2] Box in the image where the real image is
        excluding the padding.
 * Returns:
 * boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
 * class_ids: [N] Integer class IDs for each bounding box
 * scores: [N] Float probability scores of the class_id
 * masks: [height, width, num_instances] Instance masks
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, std::vector<cv::Mat>>
UnmoldDetections(at::Tensor detections,
                 at::Tensor mrcnn_mask,
                 const cv::Size& image_shape,
                 const Window& window,
                 double mask_threshold);

void VisualizeBoxes(const std::string& name,
                    int width,
                    int height,
                    at::Tensor anchors,
                    at::Tensor gt_boxes);
#endif  // IMAGEUTILS_H
