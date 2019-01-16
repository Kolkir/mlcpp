#ifndef MASKRCNN_H
#define MASKRCNN_H

#include "classifier.h"
#include "cocodataset.h"
#include "config.h"
#include "fpn.h"
#include "imageutils.h"
#include "mask.h"
#include "rpn.h"
#include "statreporter.h"

#include <torch/torch.h>
#include <memory>

class MaskRCNNImpl : public torch::nn::Module {
 public:
  MaskRCNNImpl(std::string model_dir, std::shared_ptr<Config const> config);

  /* Runs the detection pipeline.
   * images: List of images, potentially of different sizes.
   * Returns a list of dicts, one dict per image. The dict contains:
   *      rois: [N, (y1, x1, y2, x2)] detection bounding boxes
   *      class_ids: [N] int class IDs
   *      scores: [N] float probability scores for the class IDs
   *      masks: [H, W, N] instance binary masks
   */

  std::tuple<at::Tensor, at::Tensor> Detect(
      at::Tensor images,
      const std::vector<ImageMeta>& image_metas);

  /*
   * Train the model.
   * train_dataset, val_dataset: Training and validation Dataset objects.
   * learning_rate: The learning rate to train with
   * epochs: Number of training epochs. Note that previous training epochs
   *         are considered to be done alreay, so this actually determines
   *         the epochs to train in total rather than in this particaular
   *         call.
   * layers: Allows selecting wich layers to train. It can be:
   *           - A regular expression to match layer names to train
   *          - One of these predefined values:
   *            heaads: The RPN, classifier and mask heads of the network
   *            all: All the layers
   *            3+: Train Resnet stage 3 and up
   *            4+: Train Resnet stage 4 and up
   *            5+: Train Resnet stage 5 and up
   */
  void Train(CocoDataset train_dataset,
             CocoDataset val_dataset,
             double learning_rate,
             uint32_t epochs,
             std::string layers_regex);

 private:
  void Build();
  void InitializeWeights();
  void SetTrainableLayers(const std::string& layers_regex);
  std::string GetCheckpointPath(uint32_t epoch) const;
  std::tuple<float, float, float, float, float, float> TrainEpoch(
      StatReporter& reporter,
      torch::data::DataLoader<CocoDataset,
                              torch::data::samplers::RandomSampler>&
          datagenerator,
      torch::optim::SGD& optimizer,
      torch::optim::SGD& optimizer_bn,
      uint32_t steps);
  std::tuple<float, float, float, float, float, float> ValidEpoch(
      StatReporter& reporter,
      torch::data::DataLoader<CocoDataset,
                              torch::data::samplers::RandomSampler>&
          datagenerator,
      uint32_t steps);

  std::tuple<std::vector<at::Tensor>, at::Tensor, at::Tensor, at::Tensor>
  PredictRPN(at::Tensor images, int64_t proposal_count);

  std::tuple<at::Tensor, at::Tensor> PredictInference(
      at::Tensor images,
      const std::vector<ImageMeta>& image_metas);

  std::tuple<at::Tensor,
             at::Tensor,
             at::Tensor,
             at::Tensor,
             at::Tensor,
             at::Tensor,
             at::Tensor,
             at::Tensor>
  PredictTraining(at::Tensor images,
                  at::Tensor gt_class_ids,
                  at::Tensor gt_boxes,
                  at::Tensor gt_masks);

  std::tuple<torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor>
  ComputeLosses(torch::Tensor rpn_match,
                torch::Tensor rpn_bbox,
                torch::Tensor rpn_class_logits,
                torch::Tensor rpn_pred_bbox,
                torch::Tensor target_class_ids,
                torch::Tensor mrcnn_class_logits,
                torch::Tensor target_deltas,
                torch::Tensor mrcnn_bbox,
                torch::Tensor target_mask,
                torch::Tensor mrcnn_mask);
  void PrintLoss(float loss,
                 float loss_rpn_class,
                 float loss_rpn_bbox,
                 float loss_mrcnn_class,
                 float loss_mrcnn_bbox,
                 float loss_mrcnn_mask);

 private:
  std::string model_dir_;
  std::shared_ptr<Config const> config_;

  FPN fpn_{nullptr};
  torch::Tensor anchors_;
  RPN rpn_{nullptr};
  Classifier classifier_{nullptr};
  Mask mask_{nullptr};
};

TORCH_MODULE(MaskRCNN);

#endif  // MASKRCNN_H
