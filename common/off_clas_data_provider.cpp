#if defined(USE_MLU) && defined(USE_OPENCV)
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "include/off_clas_data_provider.hpp"
#include "include/off_runner.hpp"
#include "include/pipeline.hpp"
#include "include/command_option.hpp"
#include "include/common_functions.hpp"

template <class Dtype, template <class...> class Qtype>
void OffClasDataProvider<Dtype, Qtype>::Preprocess(
    const std::vector<std::pair<std::string, cv::Mat>>& sourceImages,
    std::vector<std::vector<cv::Mat> >* destImages) {
  /* Convert the input image to the input image format of the network. */
  CHECK(sourceImages.size() == destImages->size())
    << "Size of sourceImages and destImages doesn't match";
  for (int i = 0; i < sourceImages.size(); ++i) {
    auto& sourceImage = sourceImages[i].second;
    if (FLAGS_yuv) {
      cv::Mat sample_yuv;
      sourceImage.convertTo(sample_yuv, CV_8UC1);
      cv::split(sample_yuv, (*destImages)[i]);
      continue;
    }
    cv::Mat sample;
    if (sourceImage.channels() == 3 && this->inChannel_ == 1)
      cv::cvtColor(sourceImage, sample, cv::COLOR_BGR2GRAY);
    else if (sourceImage.channels() == 4 && this->inChannel_ == 1)
      cv::cvtColor(sourceImage, sample, cv::COLOR_BGRA2GRAY);
    else if (sourceImage.channels() == 4 && this->inChannel_ == 3)
      cv::cvtColor(sourceImage, sample, cv::COLOR_BGRA2BGR);
    else if (sourceImage.channels() == 1 && this->inChannel_ == 3)
      cv::cvtColor(sourceImage, sample, cv::COLOR_GRAY2BGR);
    else if (sourceImage.channels() == 3 && this->inChannel_ == 4)
      cv::cvtColor(sourceImage, sample, cv::COLOR_BGR2RGBA);
    else if (sourceImage.channels() == 1 && this->inChannel_ == 4)
      cv::cvtColor(sourceImage, sample, cv::COLOR_GRAY2RGBA);
    else
      sample = sourceImage;
    cv::Mat sample_resized;
    if (sample.size() == this->inGeometry_) {
      sample_resized = sample;
    } else if (sample.size().area() > this->inGeometry_.area()) {
      cv::Mat sample_temp;
      cv::Size image_resize;
      if (2 == this->in_resizes_.size()) {
        image_resize.height = in_resizes_[0];
        image_resize.width = in_resizes_[1];
      } else {
        int resize = this->in_resizes_.empty() ?
          std::max(256, this->inGeometry_.width) :
          in_resizes_[0];  // default 256 or assigned by user
        if (sample.rows > sample.cols) {
          image_resize.height = static_cast<int>(
              round(static_cast<float>(resize) * sample.rows / sample.cols));
          image_resize.width = resize;
        } else {
          image_resize.height = resize;
          image_resize.width = static_cast<int>(
              round(static_cast<float>(resize) * sample.cols / sample.rows));
        }
      }
      CHECK(image_resize.height >= this->inGeometry_.height)
        << "image_resize height must larger than network height";
      CHECK(image_resize.width >= this->inGeometry_.width)
        << "image_resize width must larger than network width";
      cv::resize(sample, sample_temp, image_resize, 0, 0, cv::INTER_AREA);
      const int offsetW = (image_resize.width - this->inGeometry_.width) / 2;
      const int offsetH = (image_resize.height - this->inGeometry_.height) / 2;
      sample_resized = sample_temp(cv::Rect(offsetW,
                                            offsetH,
                                            this->inGeometry_.width,
                                            this->inGeometry_.height));
    } else {
      cv::resize(sample, sample_resized, this->inGeometry_);
    }

    cv::Mat sample_float;
    if (this->inChannel_ == 3) {
      if (FLAGS_rgb) {
        cv::Mat sample_rgb;
        cv::cvtColor(sample_resized, sample_rgb, cv::COLOR_BGR2RGB);
        sample_rgb.convertTo(sample_float, CV_32FC3);
      } else {
        sample_resized.convertTo(sample_float, CV_32FC3);
      }
    } else if (this->inChannel_ == 1) {
      sample_resized.convertTo(sample_float, CV_32FC1);
    } else if (this->inChannel_ == 4) {
      if (FLAGS_input_format == 1) {
        sample_resized.convertTo(sample_float, CV_32FC4);
      } else if (FLAGS_input_format == 2 || FLAGS_input_format == 3) {
        cv::Mat sample_bgra;
        cv::cvtColor(sample_resized, sample_bgra, cv::COLOR_RGBA2BGRA);
        sample_bgra.convertTo(sample_float, CV_32FC4);
      }
    }

    cv::Mat sample_normalized;
    if (!this->meanValue_.empty() && !this->stdValue_.empty()) {
      cv::subtract(sample_float, this->mean_, sample_normalized);
      cv::divide(sample_normalized, this->std_, sample_normalized);
      if (FLAGS_scale != 1)
        sample_normalized *= FLAGS_scale;
    } else {
      sample_normalized = sample_float;
    }
    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, (*destImages)[i]);
  }
}

INSTANTIATE_OFF_CLASS(OffClasDataProvider);

#endif  // USE_MLU && USE_OPENCV
