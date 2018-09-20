#ifndef IMAGEDB_H
#define IMAGEDB_H

#include <opencv2/opencv.hpp>

struct LabelBBox {
  float x = 0;
  float y = 0;
  float width = 0;
  float height = 0;
};

struct ImageDesc {
  cv::Mat image;
  float width;
  float height;
  float scale;
  std::vector<LabelBBox> boxes;
  std::vector<float> classes;
};

class ImageDb {
 public:
  virtual ~ImageDb();
  virtual uint32_t GetImagesCount() const = 0;
  virtual ImageDesc GetImage(uint32_t index,
                             uint32_t height,
                             uint32_t width) const = 0;
};

#endif  // IMAGEDB_H
