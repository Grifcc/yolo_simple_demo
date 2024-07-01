#include "include/header.hpp"

cv::Mat imresize(const cv::Mat& src64FC1, cv::Size outSize, bool anti_aliasing_filter) {
    cv::Size2f kernelSize(4.0, 4.0);
    size_t i, j;
    int left = 0;
    double idx_src = 0.0;

    cv::Mat res_temp(src64FC1.rows, outSize.width, src64FC1.type());
    cv::Mat res(outSize, src64FC1.type());

    cv::Size2f scaleRatio;
    scaleRatio.height = static_cast<float>(outSize.height)/static_cast<float>(src64FC1.rows);
    scaleRatio.width = static_cast<float>(outSize.width)/static_cast<float>(src64FC1.cols);

    // process width
    if (scaleRatio.width >= 1.0) {
        anti_aliasing_filter = false;
    } else if (anti_aliasing_filter && (scaleRatio.width < 1.0)) {
        kernelSize.width /= scaleRatio.width;
    }
    cv::Size padSize(static_cast<int>(ceil(kernelSize.width  / 2.0)), 0);
    cv::Mat temp_pad = paddingMat(src64FC1, padSize);
    int kernel_tap = static_cast<int>(ceil(kernelSize.width));

    for (i = 0; i < src64FC1.rows; i++) {
        for (j = 0; j < outSize.width; j++) {
        idx_src = (static_cast<double>((j + 1) / scaleRatio.width))
            + 0.5 * (1.0 - 1.0 / scaleRatio.width);
        left = static_cast<int>(floor(idx_src - kernelSize.width / 2.0));

        int colEnd =  left + kernel_tap + static_cast<int>(padSize.width);
        int colStart  = left  + static_cast<int>(padSize.width);
        cv::Mat windows = temp_pad.row(i).colRange(colStart, colEnd);

        res_temp.at<double>(i, j) = iminterp1(windows, (idx_src - left),
            scaleRatio.width, kernelSize.width, anti_aliasing_filter);
        }
    }
    // process height
    if (scaleRatio.height >= 1.0) {
        anti_aliasing_filter = false;
    } else if (anti_aliasing_filter && (scaleRatio.height < 1.0)) {
        kernelSize.height /= scaleRatio.height;
    }
    padSize.height = static_cast<size_t>(ceil(kernelSize.height / 2.0));
    padSize.width = 0;
    temp_pad = paddingMat(res_temp, padSize);
    kernel_tap = static_cast<size_t>(ceil(kernelSize.height));

    for (i = 0; i < outSize.height; i++) {
        idx_src = (static_cast<double>(i + 1) / scaleRatio.height) +
                0.5 * (1.0 - 1.0 / scaleRatio.height);
        left = static_cast<int>(floor(idx_src - (kernelSize.height / 2.0)));

        for (j = 0; j < outSize.width; j++) {
            int rowEnd =  left + kernel_tap + static_cast<int>(padSize.height);
            int rowStart = left  + static_cast<int>(padSize.height);
            cv::Mat windows_temp = temp_pad.col(j).rowRange(rowStart, rowEnd);
            cv::Mat windows;
            cv::transpose(windows_temp, windows);

            res.at<double>(i, j) = iminterp1(windows, (idx_src - left),
                scaleRatio.height, kernelSize.height, anti_aliasing_filter);
        }
    }
    return res;
}
