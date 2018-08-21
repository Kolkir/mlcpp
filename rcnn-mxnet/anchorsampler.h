#ifndef ANCHORSAMPLER_H
#define ANCHORSAMPLER_H

#include <Eigen/Dense>
#include <tuple>

class AnchorSampler {
 public:
  AnchorSampler();
  std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf> Assign(
      const Eigen::MatrixXf& anchors,
      const Eigen::MatrixXf& gt_boxes,
      float im_width,
      float im_height);

 private:
  float allowed_border_ = 0.f;
  Eigen::Index num_batch_ = 256;
  Eigen::Index num_fg_ = 0;
  float fg_fraction_ = 0.5f;
  float fg_overlap_ = 0.7f;
  float bg_overlap_ = 0.3f;
};

#endif  // ANCHORSAMPLER_H
