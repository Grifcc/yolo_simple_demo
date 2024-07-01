#if defined(USE_OPENCV)
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "include/data_provider.hpp"
#include "include/pipeline.hpp"
#include "include/runner.hpp"

#include "include/command_option.hpp"
#include "include/common_functions.hpp"

bool DataProvider::imageIsEmpty() {
  if (this->imageList.empty()) {
    return true;
  }
  return false;
}

void DataProvider::readOneBatch(std::vector<std::pair<std::string, cv::Mat>>& ImageAndName) {
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
        for (int i = 0; i < this->offlineDescripter()->inN(); i++)
          ImageAndName.push_back(std::pair<std::string, cv::Mat>(file_id, img));
      } else {
        LOG(INFO) << "failed to read " << file;
      }
    }
    return;
  }

  while (image_read < this->offlineDescripter()->inN()) {
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
        LOG(ERROR) << "No image is loaded, please check your image list";
      }
    }
  }
}

void DataProvider::WrapInputLayer(std::vector<std::vector<cv::Mat> >* wrappedImages,
                                  float* inputData) {
  //  Parameter images is a vector [ ----   ] <-- images[in_n]
  //                                |
  //                                |-> [ --- ] <-- channels[3]
  // This method creates Mat objects, and places them at the
  // right offset of input stream
  int width = this->offlineDescripter()->inW();
  int height = this->offlineDescripter()->inH();
  int channels = FLAGS_yuv ? 1 : this->offlineDescripter()->inC();
  for (int i = 0; i < this->offlineDescripter()->inN(); ++i) {
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

void DataProvider::convertColor(const cv::Mat& sourceImage, cv::Mat& sample) {
  // convert sourceImage colors, where inChannel is cambricon model required input
  // channel. There might be some cases that channel of input image inconsistent with
  // input channel of cambricon model, e.g. firstconv.
  int inChannel = this->offlineDescripter()->inC();
  if (sourceImage.channels() == 3 && inChannel == 1)
    cv::cvtColor(sourceImage, sample, cv::COLOR_BGR2GRAY);
  else if (sourceImage.channels() == 4 && inChannel == 1)
    cv::cvtColor(sourceImage, sample, cv::COLOR_BGRA2GRAY);
  else if (sourceImage.channels() == 4 && inChannel == 3)
    cv::cvtColor(sourceImage, sample, cv::COLOR_BGRA2BGR);
  else if (sourceImage.channels() == 1 && inChannel == 3)
    cv::cvtColor(sourceImage, sample, cv::COLOR_GRAY2BGR);
  else if (sourceImage.channels() == 3 && inChannel == 4)
    cv::cvtColor(sourceImage, sample, cv::COLOR_BGR2RGBA);
  else if (sourceImage.channels() == 1 && inChannel == 4)
    cv::cvtColor(sourceImage, sample, cv::COLOR_GRAY2RGBA);
  else
    sample = sourceImage;
}

void DataProvider::resizeMat(const cv::Mat& sample, cv::Mat& sample_resized) {
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

void DataProvider::convertFloat(const cv::Mat& sample_resized, cv::Mat& sample_float) {
  int inChannel = this->offlineDescripter()->inC();
  if (inChannel == 3) {
    if (FLAGS_rgb) {
      cv::Mat sample_rgb;
      cv::cvtColor(sample_resized, sample_rgb, cv::COLOR_BGR2RGB);
      sample_rgb.convertTo(sample_float, CV_32FC3);
    } else {
      sample_resized.convertTo(sample_float, CV_32FC3);
    }
  } else if (inChannel == 1) {
    sample_resized.convertTo(sample_float, CV_32FC1);
  } else if (inChannel == 4) {
    if (FLAGS_input_format == 1) {
      sample_resized.convertTo(sample_float, CV_32FC4);
    } else if (FLAGS_input_format == 2 || FLAGS_input_format == 3) {
      cv::Mat sample_bgra;
      cv::cvtColor(sample_resized, sample_bgra, cv::COLOR_RGBA2BGRA);
      sample_bgra.convertTo(sample_float, CV_32FC4);
    } else {
      sample_float = sample_resized;
    }
  }
}

void DataProvider::normalizeMat(const cv::Mat& sample_float, cv::Mat& sample_normalized) {
  if (!meanValue_.empty()) {
    cv::subtract(sample_float, mean_, sample_normalized);
    if (FLAGS_scale != 1)
      sample_normalized *= FLAGS_scale;
  } else {
    sample_normalized = sample_float;
  }
}

void DataProvider::Preprocess(
    const std::vector<std::pair<std::string, cv::Mat>>& sourceImages,
    std::vector<std::vector<cv::Mat> >* destImages) {
  /* Convert the input image to the input image format of the network. */
  CHECK(sourceImages.size() == destImages->size())
    << "Size of sourceImages and destImages doesn't match";
  for (int i = 0; i < sourceImages.size(); ++i) {
    auto& sourceImage = sourceImages[i].second;
    // TODO: add yuv format
    if (FLAGS_yuv) {
      cv::Mat sample_yuv;
      sourceImage.convertTo(sample_yuv, CV_8UC1);
      cv::split(sample_yuv, (*destImages)[i]);
      continue;
    }

    // 1. convert color to BGR
    cv::Mat sample_temp_bgr;
    convertColor(sourceImage, sample_temp_bgr);

    /* 2. in convertFloat(), image is first converted from BGR to RGB, 
     *  then is converted from RGB to float. */
    cv::Mat sample_float(sourceImage.cols, sourceImage.rows, CV_32FC3);
    convertFloat(sample_temp_bgr, sample_float);

    // 3. resize and padding e.g. letterbox()
    int input_height = 640;
    int input_width = 640;
    float img_w = sourceImage.cols;
    float img_h = sourceImage.rows;

    // 3.1 calculate scale ratio (new / old)
    float img_scale = img_w < img_h ? (input_height / img_h) : (input_width / img_w);

    // 3.2 calculate new size
    int new_w = std::floor(img_w * img_scale);
    int new_h = std::floor(img_h * img_scale);
    
    // 3.3 resize to new size
    cv::Mat sample_resized;
    cv::resize(sample_float, sample_resized, cv::Size(new_w, new_h), cv::INTER_LINEAR);
    
    // 3.4 padding
    cv::Mat sample_padding(input_height,input_width,CV_32FC3,cv::Scalar(114,114,114));
    sample_resized.copyTo(sample_padding(
                                        cv::Range((static_cast<float>(input_height) - new_h) / 2,
                                                  (static_cast<float>(input_height) - new_h) / 2 + new_h),
                                        cv::Range((static_cast<float>(input_width) - new_w) / 2,
                                                  (static_cast<float>(input_width) - new_w) / 2 + new_w)));

    // 5. normalize
    // It is necessary to normalize image to a range of 0 to 1.
    cv::Mat sample_normalized(input_height,input_width,CV_32FC3);
    sample_normalized = sample_padding;

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, (*destImages)[i]);

  }
}

void DataProvider::transData() {
  int dim_order[4] = {0, 2, 3, 1};
  auto input_shape = this->offlineDescripter()->getInputShape();
  // XXX: shape should be unsigned int, but cnrt interface only accept int* pointer.
  // static_cast is needed to do the tricky
  int dim_shape[4] = {static_cast<int>(input_shape[0]), /*n*/
                      static_cast<int>(input_shape[1]), /*c*/
                      static_cast<int>(input_shape[2]), /*h*/
                      static_cast<int>(input_shape[3])}; /*w*/
  int inputDimValue[4] = {static_cast<int>(input_shape[0]), /*n*/
                          static_cast<int>(input_shape[2]), /*h*/
                          static_cast<int>(input_shape[3]), /*w*/
                          static_cast<int>(input_shape[1])}; /*c*/
  int temp_input_count = this->offlineDescripter()->inCount();
  int inputDimStride[4] = {0, 0, 0, 1};

  int inputNum = this->offlineDescripter()->inputNum();
  cnrtDataType_t* input_data_type = this->offlineDescripter()->getInputDataType();

  auto cpuCastData = reinterpret_cast<void**>(cpuCastData_);
  temp_ptrs.clear();

  for (int i = 0; i < inputNum; i++) {
    void* temp_ptr = nullptr;
    cnrtTransDataOrder(cpuData_[i],
                       CNRT_FLOAT32,
                       cpuTrans_[i],
                       4,
                       dim_shape,
                       dim_order);
    temp_ptr = cpuTrans_[i];

    if (input_data_type[i] != CNRT_FLOAT32) {
      cnrtCastDataType(cpuTrans_[i],
                       CNRT_FLOAT32,
                       cpuCastData[i],
                       input_data_type[i],
                       temp_input_count,
                       nullptr);
      temp_ptr = cpuCastData[i];
    }

    // FLAGS_input_format indicates the channel format defined in CAMBRICON MODEL.
    if (i == 0 && this->offlineDescripter()->isFirstConv()) {
      if (FLAGS_input_format == 0) {
        // input image is rgb(bgr) format
        // input_format=0 represents weight is in rgb(bgr) order
        cnrtAddDataStride(cpuCastData[0], CNRT_UINT8, cpuStrideData_.get(), 4,
                          inputDimValue, inputDimStride);
      } else if (FLAGS_input_format == 1 || FLAGS_input_format == 3) {
        // input data is in rgba(bgra) format
        // input_format=1 represents weight is in argb(abgr) order
        // input_data uses rgba and weight uses argb. To make format unified,
        // a circle shift is required.
        memcpy(cpuStrideData_.get() + 1, cpuCastData[0], temp_input_count - 1);
        cpuStrideData_.get()[0] = 0;
      } else if (FLAGS_input_format == 2) {
        // input data is in bgra(rgba) format
        // input_format=2 represents weight is in bgra(rgba) order
        memcpy(cpuStrideData_.get(), cpuCastData[0], temp_input_count);
      }
      temp_ptr = cpuStrideData_.get();
    }
    temp_ptrs.push_back(temp_ptr);
  }
}


void DataProvider::copyin(std::shared_ptr<BoxingData> input_boxing_data) {
  int inputNum = this->offlineDescripter()->inputNum();
  // retrieve data from queue
  auto mluData = input_boxing_data->getBuf();
  auto queue = input_boxing_data->getQueue();

  for (int i = 0; i < inputNum; i++) {
    if (FLAGS_async_copy) {
      cnrtMemcpyAsync(mluData.get()[i],
                      temp_ptrs[i],
                      this->offlineDescripter()->inputSizeS[i],
                      *queue,
                      CNRT_MEM_TRANS_DIR_HOST2DEV);
    } else {
      cnrtMemcpy(mluData.get()[i],
                 temp_ptrs[i],
                 this->offlineDescripter()->inputSizeS[i],
                 CNRT_MEM_TRANS_DIR_HOST2DEV);
    }
  }
}
void DataProvider::runParallel() {
  setDeviceId(deviceId_);

  prepareBuffer();
  Pipeline::waitForNotification();

  if (FLAGS_perf_mode) {
    int leftNumBatch = FLAGS_perf_mode_img_num / this->offlineDescripter()->inN();
    Timer preprocessor;
    Timer readimage;
    std::vector<std::pair<std::string, cv::Mat>> imageNameVec;
    this->readOneBatch(imageNameVec);
    readimage.log("read image by opencv ...");
    std::vector<std::vector<cv::Mat> > preprocessedImages;
    Timer prepareInput;
    this->WrapInputLayer(&preprocessedImages, reinterpret_cast<float*>(cpuData_[0]));
    this->Preprocess(imageNameVec, &preprocessedImages);
    this->transData();
    prepareInput.log("prepare input data ...");
    preprocessor.log("preprocessor time ...");

    while (leftNumBatch--) {
      auto input_boxing_data = this->offlineDescripter()->popFreeInputBoxingData(deviceId_);
      auto time_stamp = input_boxing_data->getStamp();
      Timer copyin;
      TimePoint t1 = std::chrono::high_resolution_clock::now();
      this->copyin(input_boxing_data);
      copyin.log("copyin time ...");
      TimePoint t2 = std::chrono::high_resolution_clock::now();
      time_stamp->in_start = t1;
      time_stamp->in_end = t2;

      // set boxing data
      input_boxing_data->setImageAndName(imageNameVec);
      this->offlineDescripter()->pushValidInputBoxingData(input_boxing_data, deviceId_);
      
    }
    LOG(INFO) << "DataProvider: no data ...";
    // tell runner there is no more images
    return;
  }

  while (this->imageList.size()) {
    Timer preprocessor;
    Timer readimage;
    std::vector<std::pair<std::string, cv::Mat>> imageNameVec;
    this->readOneBatch(imageNameVec);
    readimage.log("read image by opencv ...");
    std::vector<std::vector<cv::Mat> > preprocessedImages;
    Timer prepareInput;
    this->WrapInputLayer(&preprocessedImages, reinterpret_cast<float*>(cpuData_[0]));
    this->Preprocess(imageNameVec, &preprocessedImages);
    prepareInput.log("prepare input data ...");

    auto input_boxing_data = this->offlineDescripter()->popFreeInputBoxingData(deviceId_);
    auto time_stamp = input_boxing_data->getStamp();
    Timer copyin;
    TimePoint t1 = std::chrono::high_resolution_clock::now();
    this->transData();
    this->copyin(input_boxing_data);
    copyin.log("copyin time ...");
    TimePoint t2 = std::chrono::high_resolution_clock::now();
    time_stamp->in_start = t1;
    time_stamp->in_end = t2;

    // set boxing data
    input_boxing_data->setImageAndName(imageNameVec);
    this->offlineDescripter()->pushValidInputBoxingData(input_boxing_data, deviceId_);
    preprocessor.log("preprocessor time ...");
  }
  LOG(INFO) << "DataProvider: no data ...";
  // tell runner there is no more images
}
void DataProvider::SetMean() {
  if (!this->meanValue_.empty())
    SetMeanValue();
}

void DataProvider::SetMeanValue() {
  if (FLAGS_yuv) return;
  std::stringstream ss(this->meanValue_);
  std::vector<float> values;
  std::string item;
  while (getline(ss, item, ',')) {
    float value = std::atof(item.c_str());
    values.push_back(value);
  }
  int inChannel = this->offlineDescripter()->inC();
  CHECK(values.size() == 1 || values.size() == inChannel) <<
    "Specify either one mean value or as many as channels: " << inChannel;
  std::vector<cv::Mat> channels;
  for (int i = 0; i < inChannel; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(this->inGeometry_.height, this->inGeometry_.width, CV_32FC1,
                    cv::Scalar(values[i]));
    channels.push_back(channel);
  }
  cv::merge(channels, this->mean_);
}

void DataProvider::SetStd() {
  if (!this->stdValue_.empty())
    SetStdValue();
}

void DataProvider::SetStdValue() {
  if (FLAGS_yuv) return;
  std::stringstream ss(this->stdValue_);
  std::vector<float> values;
  std::string item;
  while (getline(ss, item, ',')) {
    float value = std::atof(item.c_str());
    values.push_back(value);
  }
  int inChannel = this->offlineDescripter()->inC();
  CHECK(values.size() == 1 || values.size() == inChannel) <<
    "Specify either one std value or as many as channels: " << inChannel;
  std::vector<cv::Mat> channels;
  for (int i = 0; i < inChannel; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(this->inGeometry_.height, this->inGeometry_.width, CV_32FC1,
                    cv::Scalar(values[i]));
    channels.push_back(channel);
  }
  cv::merge(channels, this->std_);
}

void DataProvider::prepareBuffer() {
  // Allocate buffers used in dataprovder to avoid malloc repeatly in for loop
  // TODO: consider whether it is proper to do shape intialization and set mean in prepareBuffer
  int inHeight = this->offlineDescripter()->inH();
  int inWidth = this->offlineDescripter()->inW();
  this->inGeometry_ = cv::Size(inWidth, inHeight);
  this->SetMean();
  this->SetStd();
  // TODO: use shared_ptr instead of raw pointer and refactor cpudata
  int inputNum = this->offlineDescripter()->inputNum();
  cpuData_ = new(void*);
  cpuTrans_ = new(void*);
  cpuCastData_ = new(void*);
  for (int i = 0; i < inputNum; ++i) {
    int input_count = this->offlineDescripter()->inCount(i);
    cpuData_[i] = new float[input_count];
    cpuTrans_[i] = new float[input_count];

    if (i == 0 && this->offlineDescripter()->isFirstConv()) {
      // The differences between input_count and inputSizeS[i] is that:
      // input_count = n * c * h * w
      // inputSizeS[i] = input_count * sizeof(data_type)
      // cpuData_ is allocated by new float[input_count]
      // cpuCastData_ is the casted data which size depends on the data type
      // Here's the tricky for firstconv. The actual input channel for firstconv
      // is 3 with data type uint8, while channel in cambricon is 4. That is,
      // in this senario, input_count = 3nhw, while inputSizeS[0]=4nhw.
      cpuCastData_[i] = new char[input_count];
      cpuStrideData_ = std::shared_ptr<uint8_t>(
        new uint8_t[this->offlineDescripter()->inputSizeS[0]],
        std::default_delete<uint8_t[]>());
    } else {
      cpuCastData_[i] = new char[this->offlineDescripter()->inputSizeS[i]];
    }
  }
}

// TODO: czr- why setDeviceId is defined here?
void setDeviceId(int deviceID) {
  unsigned devNum;
  CNRT_CHECK(cnrtGetDeviceCount(&devNum));
  if (deviceID >= 0) {
    CHECK_NE(devNum, 0) << "No device found";
    CHECK_LT(deviceID, devNum) << "Valid device count: " <<devNum;
  } else {
    LOG(FATAL) << "Invalid device number";
  }
  cnrtDev_t dev;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, deviceID));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));
}

#endif  // USE_OPENCV

