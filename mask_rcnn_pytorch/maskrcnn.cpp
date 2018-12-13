#include "maskrcnn.h"
#include "anchors.h"
#include "proposallayer.h"
#include "resnet.h"

#include <cmath>

MaskRCNNImpl::MaskRCNNImpl(std::string model_dir,
                           std::shared_ptr<Config const> config)
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
bool MaskRCNNImpl::Detect(at::Tensor images,
                          const std::vector<ImageMeta>& image_metas) {
  // Run object detection
  auto [detections, mrcnn_mask] = Predict(images, image_metas, Mode::Inference);

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
std::tuple<at::Tensor, at::Tensor> MaskRCNNImpl::Predict(
    at::Tensor images,
    const std::vector<ImageMeta>& image_metas,
    Mode mode) {
  if (mode == Mode::Inference) {
    eval();
  } else if (mode == Mode::Training) {
    // TODO: Implement
    assert(false);
  } else {
    assert(false);
  }

  // Feature extraction
  auto [p2_out, p3_out, p4_out, p5_out, p6_out] = fpn_->forward(images);

  // Note that P6 is used in RPN, but not in the classifier heads.
  std::vector<at::Tensor> rpn_feature_maps = {p2_out, p3_out, p4_out, p5_out,
                                              p6_out};
  std::vector<at::Tensor> mrcnn_feature_maps = {p2_out, p3_out, p4_out, p5_out};

  // Loop through pyramid layers
  std::vector<at::Tensor> rpn_class_logits;
  std::vector<at::Tensor> rpn_class;
  std::vector<at::Tensor> rpn_bbox;
  for (auto p : rpn_feature_maps) {
    auto [class_logits, probs, bbox] = rpn_->forward(p);
    rpn_class_logits.push_back(class_logits);
    rpn_class.push_back(probs);
    rpn_bbox.push_back(bbox);
  }

  // Generate proposals
  // Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
  // and zero padded.
  auto proposal_count = mode == Mode::Training
                            ? config_->post_nms_rois_training
                            : config_->post_nms_rois_inference;
  auto rpn_rois = ProposalLayer(
      {torch::stack(rpn_class, 1), torch::stack(rpn_bbox, 1)}, proposal_count,
      config_->rpn_nms_threshold, anchors_, config_);

  if (mode == Mode::Inference) {
  } else if (mode == Mode::Training) {
    // TODO: Implement
    assert(false);
  } else {
    assert(false);
  }
  return {};
}

// Build Mask R-CNN architecture.
void MaskRCNNImpl::Build() {
  assert(config_);

  // Image size must be dividable by 2 multiple times
  auto h = config_->image_shape[0];
  auto w = config_->image_shape[1];
  auto p = static_cast<int32_t>(std::pow(2l, 6l));
  if (static_cast<int32_t>(static_cast<double>(h) / static_cast<double>(p)) !=
          h / p ||
      static_cast<int32_t>(static_cast<double>(w) / static_cast<double>(p)) !=
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

void MaskRCNNImpl::InitializeWeights() {
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
