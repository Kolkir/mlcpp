#include "maskrcnn.h"
#include "anchors.h"
#include "resnet.h"

#include <cmath>

MaskRCNN::MaskRCNN(std::string model_dir, std::shared_ptr<Config const> config)
    : model_dir_(model_dir), config_(config) {
  Build();
  InitializeWeights();
}

/* Runs the detection pipeline.
 * images: List of images, potentially of different sizes.
 * Returns a list of dicts, one dict per image. The dict contains:
 * rois: [N, (y1, x1, y2, x2)] detection bounding boxes
 * class_ids: [N] int class IDs
 * scores: [N] float probability scores for the class IDs
 * masks: [H, W, N] instance binary masks
 */
bool MaskRCNN::Detect(const std::vector<at::Tensor>& images) {
  // Mold inputs to format expected by the neural network
  auto [molded_images, image_metas, windows] = MoldInputs(images);

  //# Convert images to torch tensor
  // molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1,
  // 2)).float()

  //# To GPU
  // if self.config.GPU_COUNT:
  //    molded_images = molded_images.cuda()

  //# Wrap in variable
  // molded_images = Variable(molded_images, volatile=True)

  //# Run object detection
  // detections, mrcnn_mask = self.predict([molded_images, image_metas],
  // mode='inference')

  //# Convert to numpy
  // detections = detections.data.cpu().numpy()
  // mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()

  //# Process detections
  // results = []
  // for i, image in enumerate(images):
  //    final_rois, final_class_ids, final_scores, final_masks =\
//        self.unmold_detections(detections[i], mrcnn_mask[i],
  //                               image.shape, windows[i])
  //    results.append({
  //        "rois": final_rois,
  //        "class_ids": final_class_ids,
  //        "scores": final_scores,
  //        "masks": final_masks,
  //    })
  // return results
  return false;
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
std::tuple<at::Tensor, at::Tensor, at::Tensor> MaskRCNN::MoldInputs(
    const std::vector<at::Tensor>& images) {
  torch::TensorList molded_images;
  torch::TensorList image_metas;
  std::vector<at::Tensor> windows;
  for (const auto& image : images) {
    // Resize image to fit the model expected size
    // TODO: move resizing to mold_image()
    //      molded_image, window, scale, padding = utils.resize_image(
    //          image,
    //          min_dim=self.config.IMAGE_MIN_DIM,
    //          max_dim=self.config.IMAGE_MAX_DIM,
    //          padding=self.config.IMAGE_PADDING)
    //      molded_image = mold_image(molded_image, self.config)
    //      # Build image_meta
    //      image_meta = compose_image_meta(
    //          0, image.shape, window,
    //          np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
    //      # Append
    //      molded_images.append(molded_image)
    //      windows.append(window)
    //      image_metas.append(image_meta)
  }
  // Pack into arrays
  return {torch::stack(molded_images), torch::stack(image_metas),
          torch::stack(windows)};
}

// Build Mask R-CNN architecture.
void MaskRCNN::Build() {
  assert(config_);

  // Image size must be dividable by 2 multiple times
  auto h = config_->image_shape[0];
  auto w = config_->image_shape[1];
  auto p = static_cast<uint32_t>(std::pow(2l, 6l));
  if (static_cast<uint32_t>(static_cast<double>(h) / static_cast<double>(p)) !=
          h / p ||
      static_cast<uint32_t>(static_cast<double>(w) / static_cast<double>(p)) !=
          w / p) {
    throw std::invalid_argument(
        "Image size must be dividable by 2 at least 6 times "
        "to avoid fractions when downscaling and upscaling."
        "For example, use 256, 320, 384, 448, 512, ... etc. ");
  }

  // Build the shared convolutional layers.
  // Bottom-up Layers
  // Returns a list of the last layers of each stage, 5 in total.
  // Don't create the thead (stage 5), so we pick the 4th item in the list.
  ResNetImpl resnet(ResNetImpl::Architecture::ResNet101, true);
  auto [C1, C2, C3, C4, C5] = resnet.GetStages();

  // Top-down Layers
  // TODO: add assert to varify feature map sizes match what's in config
  fpn_ = FPN(C1, C2, C3, C4, C5, /*out_channels*/ 256);
  register_module("fpn", fpn_);

  anchors_ = GeneratePyramidAnchors(
      config_->rpn_anchor_scales, config_->rpn_anchor_ratios,
      config_->backbone_shapes, config_->backbone_strides,
      config_->rpn_anchor_stride);

  if (config_->gpu_count > 0)
    anchors_ = anchors_.toBackend(torch::Backend::CUDA);

  // RPN
  rpn_ =
      RPN(config_->rpn_anchor_ratios.size(), config_->rpn_anchor_stride, 256);

  // FPN Classifier
  classifier_ = Classifier(256, config_->pool_size, config_->image_shape,
                           config_->num_classes);

  // FPN Mask
  mask_ = Mask(256, config_->mask_pool_size, config_->image_shape,
               config_->num_classes);

  // Fix batch norm layers
  auto set_bn_fix = [](const std::string& name, Module& m) {
    if (name.find("BatchNorm") != std::string::npos) {
      for (auto& p : m.parameters())
        p.set_requires_grad(false);
    }
  };

  apply(set_bn_fix);
}

void MaskRCNN::InitializeWeights() {
  for (auto m : modules(false)) {
    if (m->name().find("Conv2d") != std::string::npos) {
      for (auto& p : m->named_parameters()) {
        if (p.key().find("weight") != std::string::npos) {
          torch::nn::init::xavier_uniform_(p.value());
        } else if (p.key().find("bias") != std::string::npos) {
          torch::nn::init::zeros_(p.value());
        }
      }
    } else if (m->name().find("BatchNorm2d") != std::string::npos) {
      for (auto& p : m->named_parameters()) {
        if (p.key().find("weight") != std::string::npos) {
          torch::nn::init::ones_(p.value());
        }
        if (p.key().find("bias") != std::string::npos) {
          torch::nn::init::zeros_(p.value());
        }
      }
    } else if (m->name().find("Linear") != std::string::npos) {
      for (auto& p : m->named_parameters()) {
        if (p.key().find("weight") != std::string::npos) {
          torch::nn::init::normal_(p.value(), 0, 0.01);
        }
        if (p.key().find("bias") != std::string::npos) {
          torch::nn::init::zeros_(p.value());
        }
      }
    }
  }
}
