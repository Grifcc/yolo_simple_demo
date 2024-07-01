#if defined(USE_OPENCV)
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "include/data_provider.hpp"
#include "include/pipeline.hpp"

#include "include/command_option.hpp"
#include "include/common_functions.hpp"

template <class Dtype, template <class...> class Qtype>
bool DataProvider<Dtype, Qtype>::imageIsEmpty() {
  if (this->imageList.empty())
    return true;

  return false;
}

template <class Dtype, template <class...> class Qtype>
void DataProvider<Dtype, Qtype>::readOneBatch() {
  std::vector<std::pair<std::string, cv::Mat>> ImageAndName;
  std::string file_id , file;
  cv::Mat prev_image;
  int image_read = 0;
  if (FLAGS_perf_mode) {
    // only read one image and copy to a batch in performance mode
    if (!this->imageList.empty()) {
      file = file_id = this->imageList.front();
      this->imageList.pop();
      if (file.find(" ") != std::string::npos)
        file = file.substr(0, file.find(" "));
      cv::Mat img;
      if (FLAGS_yuv) {
        img = convertYuv2Mat(file, inGeometry_);
      } else {
        img = cv::imread(file, -1);
      }
      if (img.data) {
        ++image_read;
        prev_image = img;
        for (int i = 0; i < this->inNum_; i++)
          ImageAndName.push_back(std::pair<std::string, cv::Mat>(file_id, img));
      } else {
        LOG(INFO) << "failed to read " << file;
      }
    }
    this->inImageAndName_.push_back(ImageAndName);
    return;
  }

  while (image_read < this->inNum_) {
    if (!this->imageList.empty()) {
      file = file_id = this->imageList.front();
      this->imageList.pop();
      if (file.find(" ") != std::string::npos)
        file = file.substr(0, file.find(" "));
      cv::Mat img;
      if (FLAGS_yuv) {
        img = convertYuv2Mat(file, inGeometry_);
      } else {
        img = cv::imread(file, -1);
      }
      if (img.data) {
        ++image_read;
        prev_image = img;
        ImageAndName.push_back(std::pair<std::string, cv::Mat>(file_id, img));

      } else {
        LOG(INFO) << "failed to read " << file;
      }
    } else {
      if (image_read) {
        cv::Mat img;
        ++image_read;
        prev_image.copyTo(img);
        ImageAndName.push_back(std::pair<std::string, cv::Mat>("null", img));
      } else {
        // if the que is empty and no file has been read, no more runs
        return;
      }
    }
  }

  this->inImageAndName_.push_back(ImageAndName);
}

template <class Dtype, template <class...> class Qtype>
void DataProvider<Dtype, Qtype>::preRead() {
  const float imageList_size = this->imageList.size();
  const int inImageAndName_size = ceil(imageList_size / this->inNum_);
  this->inImageAndName_.reserve(inImageAndName_size);
  while (this->imageList.size()) {
    this->readOneBatch();
  }
}

template <class Dtype, template <class...> class Qtype>
void DataProvider<Dtype, Qtype>::WrapInputLayer(std::vector<std::vector<cv::Mat> >* wrappedImages,
                                  float* inputData) {
  //  Parameter images is a vector [ ----   ] <-- images[in_n]
  //                                |
  //                                |-> [ --- ] <-- channels[3]
  // This method creates Mat objects, and places them at the
  // right offset of input stream
  int width = this->runner_->w();
  int height = this->runner_->h();
  int channels = FLAGS_yuv ? 1 : this->runner_->c();
  for (int i = 0; i < this->runner_->n(); ++i) {
    wrappedImages->push_back(std::vector<cv::Mat> ());
    for (int j = 0; j < channels; ++j) {
      if (FLAGS_yuv) {
        cv::Mat channel(height, width, CV_8UC1, reinterpret_cast<char*>(inputData));
        (*wrappedImages)[i].push_back(channel);
        inputData += width * height / 4;
      } else {
        cv::Mat channel(height, width, CV_32FC1, inputData);
        (*wrappedImages)[i].push_back(channel);
        inputData += width * height;
      }
    }
  }
}

template <class Dtype, template <class...> class Qtype>
void DataProvider<Dtype, Qtype>::resizeMat(const cv::Mat& sample, cv::Mat& sample_resized) {
    if (sample.size() == inGeometry_) {
      sample_resized = sample;
      return;
    }
    // According to OpenCV doc, to shrink an image, the result generally look best
    // with INTER_AREA. Otherwise use INTER_LINEAR. Meanwhile, Pytorch python frontend
    // uses PIL instead of OpenCV, that's why interpolation methods matters.
    if (sample.size().area() > inGeometry_.area()) {
      auto interpolation = FLAGS_interpolation ?  cv::INTER_AREA : cv::INTER_LINEAR;
      if (FLAGS_preprocess_method) {
        sample_resized = scaleResizeCrop(sample, inGeometry_);
      } else {
        cv::resize(sample, sample_resized, inGeometry_, 0, 0, interpolation);
      }
    } else {
      cv::resize(sample, sample_resized, inGeometry_);
    }
}

template <class Dtype, template <class...> class Qtype>
void DataProvider<Dtype, Qtype>::normalizeMat(const cv::Mat& sample_float, cv::Mat& sample_normalized) {
  if (!meanValue_.empty()) {
    cv::subtract(sample_float, mean_, sample_normalized);
    if (FLAGS_scale != 1)
      sample_normalized *= FLAGS_scale;
  } else {
    sample_normalized = sample_float;
  }
}

template <class Dtype, template <class...> class Qtype>
void DataProvider<Dtype, Qtype>::Preprocess(
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
    if (FLAGS_ycbcr) {
      cv::Mat imageY = resizeAndConvertBgrToY(sourceImages[i], inGeometry_, false);
      cv::split(imageY, (*destImages)[i]);
      continue;
    }
    cv::Mat sample;
    if (sourceImage.channels() == 3 && inChannel_ == 1)
      cv::cvtColor(sourceImage, sample, cv::COLOR_BGR2GRAY);
    else if (sourceImage.channels() == 4 && inChannel_ == 1)
      cv::cvtColor(sourceImage, sample, cv::COLOR_BGRA2GRAY);
    else if (sourceImage.channels() == 4 && inChannel_ == 3)
      cv::cvtColor(sourceImage, sample, cv::COLOR_BGRA2BGR);
    else if (sourceImage.channels() == 1 && inChannel_ == 3)
      cv::cvtColor(sourceImage, sample, cv::COLOR_GRAY2BGR);
    else if (sourceImage.channels() == 3 && inChannel_ == 4)
      cv::cvtColor(sourceImage, sample, cv::COLOR_BGR2RGBA);
    else if (sourceImage.channels() == 1 && inChannel_ == 4)
      cv::cvtColor(sourceImage, sample, cv::COLOR_GRAY2RGBA);
    else
      sample = sourceImage;
    cv::Mat sample_resized;
    resizeMat(sample, sample_resized);
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
      if ((FLAGS_rgb && FLAGS_input_format == 1) || (!FLAGS_rgb &&
          (FLAGS_input_format == 2 || FLAGS_input_format == 3))) {
        sample_resized.convertTo(sample_float, CV_32FC4);
      } else if ((FLAGS_rgb && (FLAGS_input_format == 2 || FLAGS_input_format == 3))
                 || (!FLAGS_rgb && FLAGS_input_format == 1)) {
        cv::Mat sample_bgra;
        cv::cvtColor(sample_resized, sample_bgra, cv::COLOR_RGBA2BGRA);
        sample_bgra.convertTo(sample_float, CV_32FC4);
      }
    }

    cv::Mat sample_normalized;
    normalizeMat(sample_float, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, (*destImages)[i]);
  }
}

template <class Dtype, template <class...> class Qtype>
void DataProvider<Dtype, Qtype>::SetStd() {
  if (!this->stdValue_.empty())
    SetStdValue();
}

template <class Dtype, template <class...> class Qtype>
void DataProvider<Dtype, Qtype>::SetStdValue() {
  if (FLAGS_yuv) return;
  cv::Scalar channel_std;
  std::stringstream ss(this->stdValue_);
  std::vector<float> values;
  std::string item;
  while (getline(ss, item, ',')) {
    float value = std::atof(item.c_str());
    values.push_back(value);
  }
  CHECK(values.size() == 1 || values.size() == this->inChannel_) <<
    "Specify either one std value or as many as channels: " << inChannel_;
  std::vector<cv::Mat> channels;
  for (int i = 0; i < inChannel_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(this->inGeometry_.height, this->inGeometry_.width, CV_32FC1,
                    cv::Scalar(values[i]));
    channels.push_back(channel);
  }
  cv::merge(channels, this->std_);
}

template <class Dtype, template <class...> class Qtype>
void DataProvider<Dtype, Qtype>::SetMean() {
  if (!this->meanValue_.empty())
    SetMeanValue();
}

template <class Dtype, template <class...> class Qtype>
void DataProvider<Dtype, Qtype>::SetMeanValue() {
  if (FLAGS_yuv) return;
  cv::Scalar channel_mean;
  std::stringstream ss(this->meanValue_);
  std::vector<float> values;
  std::string item;
  while (getline(ss, item, ',')) {
    float value = std::atof(item.c_str());
    values.push_back(value);
  }
  CHECK(values.size() == 1 || values.size() == this->inChannel_) <<
    "Specify either one mean value or as many as channels: " << inChannel_;
  std::vector<cv::Mat> channels;
  for (int i = 0; i < inChannel_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(this->inGeometry_.height, this->inGeometry_.width, CV_32FC1,
                    cv::Scalar(values[i]));
    channels.push_back(channel);
  }
  cv::merge(channels, this->mean_);
}

INSTANTIATE_ALL_CLASS(DataProvider);

#endif  // USE_OPENCV
