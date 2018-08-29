#ifndef ANCHORGENERATOR_H
#define ANCHORGENERATOR_H

#include "params.h"

#include <Eigen/Dense>
#include <vector>

class AnchorGenerator {
 public:
  AnchorGenerator(const Params& params);
  Eigen::MatrixXf Generate(uint32_t width, uint32_t height) const;

 private:
  float stride_;
  Eigen::Array3f scales_;
  Eigen::Array3f ratios_;
  Eigen::Index num_anchors_;
  Eigen::MatrixXf base_anchors_;
};

#endif  // ANCHORGENERATOR_H
