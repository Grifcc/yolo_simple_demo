#include "include/header.hpp"

cv::Mat paddingMat(const cv::Mat& src, cv::Size size) {
    cv::Size outSize(src.cols+2*size.width, src.rows+2*size.height);
    cv::Mat res(outSize, src.type());
    cv::copyMakeBorder(src, res, size.height, size.height,
                        size.width, size.width, cv::BORDER_REFLECT);
    return res;
}

double interp_kernel_func(double x) {
    double res = 0.0;
    double absx = fabs(x);
    double absx2 = absx * absx;
    double absx3 = absx * absx2;
    /* See Keys, "Cubic Convolution Interpolation for Digital Image
    Processing, " IEEE Transactions on Acoustics, Speech, and Signal
    Processing, Vol.ASSP - 29, No. 6, December 1981, p. 1155. */
    res = (1.5*absx3 - 2.5*absx2 + 1.0) * (absx <= 1) +
    (-0.5*absx3 + 2.5*absx2 - 4.0 * absx + 2.0) * ((1 < absx) && (absx <= 2));

    return res;
}


double iminterp1(const cv::Mat& windows, double idx_fraction,
                 double scaling_ratio, double interp_kernel_taps,
                 bool anti_aliasing_filter) {
  size_t i = 0;
  size_t kernel_taps = static_cast<size_t>(ceil(interp_kernel_taps));
  double *interp_kernel_window = static_cast<double*>(calloc(kernel_taps, sizeof(double)));
  double cum_sum_interp_kernel_window = 0.0;
  double interesting_pst = 0.0;
  double initial_pst = 0.0;
  double res = 0.0;

  initial_pst = (idx_fraction - 1.0);
  for (i = 0; i < kernel_taps; i++) {
    interesting_pst = initial_pst - static_cast<double>(i);

    if (anti_aliasing_filter) {
      interesting_pst *= scaling_ratio;
    }
    interp_kernel_window[i] = interp_kernel_func(interesting_pst);
    if (anti_aliasing_filter) {
      interp_kernel_window[i] *= scaling_ratio;
    }
    cum_sum_interp_kernel_window += interp_kernel_window[i];
  }

  for (i = 0; i < kernel_taps; i++) {
    res += windows.at<double>(0, i) * (interp_kernel_window[i] / cum_sum_interp_kernel_window);
  }

  free(interp_kernel_window);
  return res;
}
