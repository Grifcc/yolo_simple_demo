#ifndef CATCH_EXAMPLES_OFFLINE_COMMON_BICUBICRESIZE_INCLUDE_HEADER_HPP_
#define CATCH_EXAMPLES_OFFLINE_COMMON_BICUBICRESIZE_INCLUDE_HEADER_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <memory.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


typedef struct double_matrix_size {
  double **buf;
  size_t row;
  size_t col;
} double_buffer;

void error(std::string error_statement);


cv::Mat imresize(const cv::Mat& src64FC1, cv::Size outSize, bool anti_aliasing_filter);
cv::Mat paddingMat(const cv::Mat& src, cv::Size size);
double iminterp1(const cv::Mat& windows, double idx_fraction,
                 double scaling_ratio, double interp_kernel_taps,
                 bool anti_aliasing_filter);
double interp_kernel_func(double x);

#endif  //  CATCH_EXAMPLES_OFFLINE_COMMON_BICUBICRESIZE_INCLUDE_HEADER_HPP_
