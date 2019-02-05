#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

/* Base Configuration Class
 * Don't use this class directly. Instead, sub-class it and override
 * the configurations you need to change.
 */
class Config {
 public:
  Config();
  void UpdateSettings();

  std::string name;  // Override in sub-classes

  // Path to pretrained imagenet model
  std::string imagenet_model_path;

  // NUMBER OF GPUs to use. For CPU use 0
  uint32_t gpu_count = 1;

  // Number of images to train with on each GPU. A 12GB GPU can typically
  // handle 2 images of 1024x1024px.
  // Adjust based on your GPU memory and image sizes. Use the highest
  // number that your GPU can handle for best performance.
  uint32_t images_per_gpu = 1;

  // Number of training steps per epoch
  // This doesn't need to match the size of the training set. Tensorboard
  // updates are saved at the end of each epoch, so setting this to a
  // smaller number means getting more frequent TensorBoard updates.
  // Validation stats are also calculated at each epoch end and they
  // might take a while, so don't set this too small to avoid spending
  // a lot of time on validation stats.
  uint32_t steps_per_epoch = 1000;

  // Number of validation steps to run at the end of every training epoch.
  // A bigger number improves accuracy of validation stats, but slows
  // down the training.
  uint32_t validation_steps = 50;

  // The strides of each layer of the FPN Pyramid. These values
  // are based on a Resnet101 backbone.
  std::vector<float> backbone_strides = {4, 8, 16, 32, 64};
  std::vector<std::pair<float, float>> backbone_shapes;

  // Number of classification classes (including background)
  uint32_t num_classes = 1;  // Override in sub-classes

  // Length of square anchor side in pixels
  std::vector<float> rpn_anchor_scales = {32, 64, 128, 256, 512};

  // Ratios of anchors at each cell (width/height)
  // A value of 1 represents a square anchor, and 0.5 is a wide anchor
  std::vector<float> rpn_anchor_ratios = {0.5f, 1.f, 2.f};

  // Anchor stride
  // If 1 then anchors are created for each cell in the backbone feature map.
  // If 2, then anchors are created for every other cell, and so on.
  uint32_t rpn_anchor_stride = 1;

  // Non-max suppression threshold to filter RPN proposals.
  // You can reduce this during training to generate more propsals.
  float rpn_nms_threshold = 0.5f;  // 0.7

  // You can reduce this during training to generate more positive anchors.
  float anchor_iou_max_threshold = 0.5f;  // 0.7 original

  // How many anchors per image to use for RPN training
  int32_t rpn_train_anchors_per_image = 256;

  // ROIs kept after non-maximum supression (training and inference)
  int64_t post_nms_rois_training = 2000;
  int64_t post_nms_rois_inference = 1000;

  // If enabled, resizes instance masks to a smaller size to reduce
  // memory load. Recommended when using high-resolution images.
  bool use_mini_mask = true;
  std::vector<int32_t> mini_mask_shape = {
      56, 56};  // (height, width) of the mini-mask

  // Input image resing
  // Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
  // the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
  // be satisfied together the IMAGE_MAX_DIM is enforced.
  int32_t image_min_dim = 800;
  int32_t image_max_dim = 1024;
  // If True, pad images with zeros such that they're (max_dim by max_dim)
  bool image_padding = true;  // currently, the False option is not supported

  // Image mean (RGB)
  std::vector<double> mean_pixel = {123.7, 116.8, 103.9};

  // Number of ROIs per image to feed to classifier/mask heads
  // The Mask RCNN paper uses 512 but often the RPN doesn't generate
  // enough positive proposals to fill this and keep a positive:negative
  // ratio of 1:3. You can increase the number of proposals by adjusting
  // the RPN NMS threshold.
  uint32_t train_rois_per_image = 200;

  // Percent of positive ROIs used to train classifier/mask heads
  float roi_positive_ratio = 0.33f;

  // Pooled ROIs
  uint32_t pool_size = 7;
  uint32_t mask_pool_size = 14;
  std::vector<int32_t> mask_shape = {28, 28};

  // Maximum number of ground truth instances to use in one image
  uint32_t max_gt_instances = 100;

  // Bounding box refinement standard deviation for RPN and final detections.
  std::vector<float> rpn_bbox_std_dev = {0.1f, 0.1f, 0.2f, 0.2f};
  std::vector<float> bbox_std_dev = {0.1f, 0.1f, 0.2f, 0.2f};

  // Max number of final detections
  int64_t detection_max_instances = 100;

  // Minimum probability value to accept a detected instance
  // ROIs below this threshold are skipped
  float detection_min_confidence = 0.7f;

  // Non-maximum suppression threshold for detection
  float detection_nms_threshold = 0.3f;

  // Learning rate and momentum
  // The Mask RCNN paper uses lr=0.02, but it can cause
  // weights to explode. Likely due to differences in optimzer
  // implementation.
  double learning_rate = 0.001;
  double learning_momentum = 0.9;

  // Weight decay regularization
  double weight_decay = 0.0001;

  // Use RPN ROIs or externally generated ROIs for training
  // Keep this True for most situations. Set to False if you want to train
  // the head branches on ROI generated by code rather than the ROIs from
  // the RPN. For example, to debug the classifier head without having to
  // train the RPN.
  // bool use_rpn_rois = true;

  // Effective batch size
  uint32_t batch_size = 0;

  // input image size
  std::vector<int32_t> image_shape;
};

#endif  // CONFIG_H
