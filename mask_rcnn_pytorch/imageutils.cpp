#include "imageutils.h"

cv::Mat LoadImage(const std::string path) {
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
  return image;
}

at::Tensor CvImageToTensor(const cv::Mat& image) {
  // Idea taken from https://github.com/pytorch/pytorch/issues/12506
  // we have to split the interleaved channels
  cv::Mat bgr[3];
  cv::split(image, bgr);
  cv::Mat channelsConcatenated;
  vconcat(bgr[0], bgr[1], channelsConcatenated);
  vconcat(channelsConcatenated, bgr[2], channelsConcatenated);

  cv::Mat channelsConcatenatedFloat;
  channelsConcatenated.convertTo(channelsConcatenatedFloat, CV_32FC3);

  std::vector<int64_t> dims{1, static_cast<int64_t>(image.channels()),
                            static_cast<int64_t>(image.rows),
                            static_cast<int64_t>(image.cols)};

  at::TensorOptions options(at::kFloat);
  at::Tensor tensor_image =
      torch::from_blob(channelsConcatenated.data, at::IntList(dims), options);
  return tensor_image;
}

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
 *  padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
 */
std::tuple<cv::Mat, Window, float, Padding> ResizeImage(cv::Mat image,
                                                        int32_t min_dim,
                                                        int32_t max_dim,
                                                        bool do_padding) {
  // Default window (y1, x1, y2, x2) and default scale == 1.
  auto h = image.rows;
  auto w = image.cols;
  Window window{0, 0, h, w};
  Padding padding;
  float scale = 1.f;

  // Scale?
  if (min_dim != 0) {
    // Scale up but not down
    scale = std::max(1.f, static_cast<float>(min_dim) / std::min(h, w));
  }

  // Does it exceed max dim?
  if (max_dim != 0) {
    auto image_max = std::max(h, w);
    if (std::round(image_max * scale) > max_dim)
      scale = static_cast<float>(max_dim) / image_max;
  }
  // Resize image and mask
  if (scale != 1.f) {
    cv::resize(image, image,
               cv::Size(static_cast<int>(std::round(h * scale)),
                        static_cast<int>(std::round(w * scale))));
  }
  // Need padding?
  if (do_padding) {
    // Get new height and width
    h = image.rows;
    w = image.cols;
    auto top_pad = (max_dim - h) / 2;
    auto bottom_pad = max_dim - h - top_pad;
    auto left_pad = (max_dim - w) / 2;
    auto right_pad = max_dim - w - left_pad;
    cv::copyMakeBorder(image, image, top_pad, bottom_pad, left_pad, right_pad,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    padding = {top_pad, bottom_pad, left_pad, right_pad, 0, 0};
    window = {top_pad, left_pad, h + top_pad, w + left_pad};
  }
  return {image, window, scale, padding};
}

/*
 * Takes RGB images with 0-255 values and subtraces
 * the mean pixel and converts it to float. Expects image
 * colors in RGB order.
 */
cv::Mat MoldImage(cv::Mat image, const Config& config) {
  image.convertTo(image, CV_32F);
  cv::Scalar mean(config.mean_pixel[0], config.mean_pixel[1],
                  config.mean_pixel[2]);
  image -= mean;
  return image;
}

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
    const Config& config) {
  std::vector<at::Tensor> molded_images;
  std::vector<ImageMeta> image_metas;
  std::vector<Window> windows;
  for (const auto& image : images) {
    // Resize image to fit the model expected size
    auto [molded_image, window, scale, padding] =
        ResizeImage(image, config.image_min_dim, config.image_max_dim,
                    config.image_padding);
    molded_image = MoldImage(molded_image, config);
    // Build image_meta
    ImageMeta image_meta{0, image.rows, image.cols, window,
                         std::vector<int32_t>(config.num_classes, 0)};

    // To tensor
    auto img_t = CvImageToTensor(molded_image);

    // Append
    molded_images.push_back(img_t);
    windows.push_back(window);
    image_metas.push_back(image_meta);
  }
  // Pack into arrays
  auto tensor_images = torch::stack(molded_images);
  // To GPU
  if (config.gpu_count > 0)
    tensor_images = tensor_images.cuda();

  return {tensor_images, image_metas, windows};
}
