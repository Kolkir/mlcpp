#include "catch.hpp"

#include "../nnutils.h"

TEST_CASE("SamePad2d equal kernel", "[nnutils]") {
  at::Tensor x = torch::ones({1, 1, 2, 2});
  SamePad2d func(/*kernel*/ 2, /*stride*/ 1);
  at::Tensor y = func->forward(x);
  REQUIRE(y.size(2) == 3);
  REQUIRE(y.size(3) == 3);
}

TEST_CASE("SamePad2d not equal kernel", "[nnutils]") {
  at::Tensor x = torch::ones({1, 1, 13, 13});
  SamePad2d func(/*kernel*/ 6, /*stride*/ 5);
  at::Tensor y = func->forward(x);
  auto y_data = y.accessor<float, 4>();
  REQUIRE(y.size(2) == 16);
  REQUIRE(y.size(3) == 16);

  REQUIRE(y_data[0][0][7][0] == Approx(0));
  REQUIRE(y_data[0][0][7][1] == Approx(1));

  REQUIRE(y_data[0][0][5][13] == Approx(1));
  REQUIRE(y_data[0][0][5][14] == Approx(0));
  REQUIRE(y_data[0][0][5][15] == Approx(0));

  REQUIRE(y_data[0][0][0][10] == Approx(0));
  REQUIRE(y_data[0][0][1][10] == Approx(1));

  REQUIRE(y_data[0][0][13][3] == Approx(1));
  REQUIRE(y_data[0][0][14][3] == Approx(0));
  REQUIRE(y_data[0][0][15][3] == Approx(0));
}

TEST_CASE("upsample size check", "[nnutils]") {
  at::Tensor x1 = torch::ones({3});
  at::Tensor x2 = torch::ones({3, 3});
  at::Tensor x3 = torch::ones({1, 3, 3});
  at::Tensor x4 = torch::ones({1, 2, 3, 3});
  at::Tensor x5 = torch::ones({1, 2, 2, 3, 3});
  at::Tensor x6 = torch::ones({1, 2, 2, 2, 3, 3});
  REQUIRE_THROWS(upsample(x1, 2));
  REQUIRE_THROWS(upsample(x2, 2));
  CHECK_NOTHROW(upsample(x3, 2));
  CHECK_NOTHROW(upsample(x4, 2));
  CHECK_NOTHROW(upsample(x5, 2));
  REQUIRE_THROWS(upsample(x6, 2));
}

TEST_CASE("upsample result check", "[nnutils]") {
  at::Tensor l = torch::linspace(0, 2, 3);
  l = l.reshape({1, 1, 3});
  auto x = at::cat({l, l, l}, 1);
  at::Tensor y;
  CHECK_NOTHROW(y = upsample(x, 2));
  auto y_data = y.accessor<float, 3>();
  REQUIRE(y_data[0][1][0] == Approx(0));
  REQUIRE(y_data[0][2][5] == Approx(2));
}
