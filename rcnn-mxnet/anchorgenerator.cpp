#include "anchorgenerator.h"

#include <iostream>
#include <tuple>

namespace {

auto ltwh_to_whcxcy(const Eigen::Array4f& anchor) {
  auto w = anchor[2] - anchor[0] + 1;
  auto h = anchor[3] - anchor[1] + 1;
  auto cx = anchor[0] + 0.5f * (w - 1);
  auto cy = anchor[1] + 0.5f * (h - 1);
  return std::make_tuple(w, h, cx, cy);
}

auto whcxcy_to_anchors(const Eigen::ArrayXf& w,
                       const Eigen::ArrayXf& h,
                       float cx,
                       float cy) {
  auto nrows = w.size();
  Eigen::MatrixX4f result(nrows, 4);
  result.col(0) = (cx - 0.5f * (w - 1));  // lt - x
  result.col(1) = (cy - 0.5f * (h - 1));  // lt - y
  result.col(2) = (cx + 0.5f * (w - 1));  // br - x
  result.col(3) = (cy + 0.5f * (h - 1));  // br - y
  return result;
}

auto meshgrid(const Eigen::ArrayXf& x, const Eigen::ArrayXf& y) {
  auto nx = x.size();
  auto ny = y.size();
  Eigen::MatrixXf X(ny, nx);
  Eigen::MatrixXf Y(ny, nx);
  for (Eigen::Index i = 0; i < ny; ++i) {
    X.row(i) = x.transpose();
  }
  for (Eigen::Index j = 0; j < nx; ++j) {
    Y.col(j) = y;
  }
  return std::make_tuple(X, Y);
}

}  // namespace

AnchorGenerator::AnchorGenerator() {
  num_anchors_ = scales_.size() * ratios_.size();
  base_anchors_.resize(num_anchors_, 4);

  Eigen::Array4f base_anchor{1, 1, stride_, stride_};
  base_anchor -= Eigen::Array4f::Ones();
  // enumerate ratios
  auto [w, h, cx, cy] = ltwh_to_whcxcy(base_anchor);
  auto size = w * h;

  auto size_ratios = Eigen::ArrayXf::Constant(ratios_.size(), size) / ratios_;
  // important to use Eigen::Array type, otherwise the result will be different
  Eigen::ArrayXf ws = size_ratios.sqrt().round();
  Eigen::ArrayXf hs = (ws * ratios_).round();
  auto anchors = whcxcy_to_anchors(ws, hs, cx, cy);
  // enumerate scales
  Eigen::Index k = 0;
  for (Eigen::Index i = 0; i < anchors.rows(); ++i) {
    auto [rw, rh, rcx, rcy] = ltwh_to_whcxcy(anchors.row(i).array());
    Eigen::ArrayXf rws = rw * scales_;
    Eigen::ArrayXf rhs = rh * scales_;
    auto anchors = whcxcy_to_anchors(rws, rhs, rcx, rcy);
    for (Eigen::Index j = 0; j < anchors.rows(); ++j) {
      base_anchors_.row(k++) = anchors.row(j);
    }
  }
}

Eigen::MatrixXf AnchorGenerator::Generate(uint32_t width,
                                          uint32_t height) const {
  Eigen::ArrayXf shift_x = Eigen::ArrayXf::LinSpaced(width, 0, width) * stride_;
  Eigen::ArrayXf shift_y =
      Eigen::ArrayXf::LinSpaced(height, 0, height) * stride_;
  auto [shift_X, shift_Y] = meshgrid(shift_x, shift_y);
  Eigen::Map<Eigen::RowVectorXf> flat_x(shift_X.data(), shift_X.size());
  Eigen::Map<Eigen::RowVectorXf> flat_y(shift_Y.data(), shift_Y.size());
  Eigen::MatrixXf shifts(4, flat_x.size());
  shifts << flat_x, flat_y, flat_x, flat_y;
  shifts.transposeInPlace();
  auto ncells = shifts.rows();
  Eigen::MatrixXf all_anchors = base_anchors_.colwise().replicate(ncells) +
                                shifts.colwise().replicate(num_anchors_);
  return all_anchors;
}
