#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H

#include "bbox.h"

#include <opencv2/opencv.hpp>

#include <string>
#include <tuple>
#include <vector>

std::pair<cv::Mat, float> LoadImage(const std::string& file_name,
                                    uint32_t short_side,
                                    uint32_t long_side);

// resize image with constraint proportions and pad with zero
std::tuple<cv::Mat, float> LoadImageFitSize(const std::string& file_name,
                                            uint32_t height,
                                            uint32_t width);

std::vector<float> CVToMxnetFormat(const cv::Mat& img);

void ShowResult(const std::vector<Detection>& detection,
                const std::string& file_name,
                const std::string& out_file_name,
                const std::vector<std::string>& classes,
                float thresh = 0.7f);

#endif  // IMAGEUTILS_H
