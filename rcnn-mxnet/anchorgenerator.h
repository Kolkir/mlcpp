#ifndef ANCHORGENERATOR_H
#define ANCHORGENERATOR_H

#include <Eigen/Dense>
#include <vector>

class AnchorGenerator {
 public:
  AnchorGenerator();
  Eigen::MatrixXf Generate(uint32_t width, uint32_t height) const;

 private:
  float stride_{16};
  Eigen::Array3f scales_{8, 16, 32};
  Eigen::Array3f ratios_{0.5f, 1.f, 2.f};
  Eigen::Index num_anchors_{0};
  Eigen::MatrixXf base_anchors_;
};

#endif  // ANCHORGENERATOR_H
