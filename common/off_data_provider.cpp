#if defined(USE_MLU) && defined(USE_OPENCV)
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "include/off_data_provider.hpp"
#include "include/pipeline.hpp"
#include "include/command_option.hpp"
#include "include/common_functions.hpp"

template <class Dtype, template <class...> class Qtype>
void OffDataProvider<Dtype, Qtype>::allocateMemory(int queueLength) {
  OffRunner<Dtype, Qtype> *off_runner = static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);
  for (int i = 0; i < queueLength; i++) {
    // this queue is for async copy
    cnrtQueue_t queue;
    if (FLAGS_async_copy) {
      CHECK(cnrtCreateQueue(&queue) == CNRT_RET_SUCCESS)
        << "CNRT create queue for async copy failed, thread_id" << off_runner->threadId();
    }

    int inputNum = off_runner->inBlobNum();
    int outputNum = off_runner->outBlobNum();
    auto inputMluPtrS = MakeArray<Dtype>(inputNum);
    auto outputMluPtrS = MakeArray<Dtype>(outputNum);

    cnrtGetInputDataSize(&(off_runner->inputSizeS), &inputNum, off_runner->function());
    for (int i = 0; i < inputNum; i++) {
      cnrtMalloc(&(inputMluPtrS.get()[i]), off_runner->inputSizeS[i]);
    }

    cnrtGetOutputDataSize(&(off_runner->outputSizeS), &outputNum, off_runner->function());
    for (int i = 0; i < outputNum; i++) {
      cnrtMalloc(&(outputMluPtrS.get()[i]), off_runner->outputSizeS[i]);
    }
    off_runner->pushFreeInputData(inputMluPtrS);
    if (FLAGS_async_copy) {
      off_runner->pushFreeAsyncCopyQueue(queue);
    }
    off_runner->pushFreeOutputData(outputMluPtrS);

    if (FLAGS_async_copy) {
      pushAsyncCopyQueueVector(queue);
    }

    // timestamp (smart pointer) for recording starting time & ending time
    auto timestamp = std::make_shared<InferenceTimeTrace>();
    off_runner->pushFreeInputTimeTraceData(timestamp);
  }

  cpuData_ = new(Dtype);
  cpuData_[0] = new float[input_count];
  cpuTempData_ = new(Dtype);
  cpuTempData_[0] = new char[off_runner->inputSizeS[0]];
  cpuTrans_ = new(Dtype);
  cpuTrans_[0] = new float[input_count];
  firstConvData_ = new(Dtype);
  firstConvData_[0] = new char[input_count];
}

template <class Dtype, template <class...> class Qtype>
void OffDataProvider<Dtype, Qtype>::init() {
  OffRunner<Dtype, Qtype> *off_runner = static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);
  setDeviceId(off_runner->deviceId());

  this->inNum_ = off_runner->n();
  this->inChannel_ = off_runner->c();
  this->inHeight_ = off_runner->h();
  this->inWidth_ = off_runner->w();
  this->inGeometry_ = cv::Size(this->inWidth_, this->inHeight_);
  this->SetMean();
  this->SetStd();

  this->inBlobNum_ = off_runner->inBlobNum();
  CNRT_CHECK(cnrtGetInputDataType(&this->input_data_type,
                                  &this->inBlobNum_, off_runner->function()));
  this->input_count = off_runner->n() * off_runner->c() * off_runner->h() * off_runner->w();
  this->dim_order[0] = 0;
  this->dim_order[1] = 2;
  this->dim_order[2] = 3;
  this->dim_order[3] = 1;

  this->dim_shape[0] = off_runner->n();
  this->dim_shape[1] = off_runner->c();
  this->dim_shape[2] = off_runner->h();
  this->dim_shape[3] = off_runner->w();
  this->inputDimValue[0] = off_runner->n();
  this->inputDimValue[1] = off_runner->h();
  this->inputDimValue[2] = off_runner->w();
  this->inputDimValue[3] = off_runner->c();

  this->inputDimStride[0] = 0;
  this->inputDimStride[1] = 0;
  this->inputDimStride[2] = 0;
  this->inputDimStride[3] = 1;

  allocateMemory(FLAGS_fifosize);
  Pipeline<Dtype, Qtype>::waitForNotification();
}

template <class Dtype, template <class...> class Qtype>
void OffDataProvider<Dtype, Qtype>::readInputData() {
  Timer readimage;
  this->inImageAndName_.clear();
  this->readOneBatch();
  if (this->inImageAndName_.empty()) return;
  this->imageNameVec.clear();
  this->imageNameVec = this->inImageAndName_[0];
  readimage.log("read image by opencv ...");
}

template <class Dtype, template <class...> class Qtype>
void OffDataProvider<Dtype, Qtype>::prepareInput() {
  std::vector<std::vector<cv::Mat> > preprocessedImages;
  Timer prepareInput;
  this->WrapInputLayer(&preprocessedImages, reinterpret_cast<float*>(cpuData_[0]));
  this->Preprocess(this->imageNameVec, &preprocessedImages);
  prepareInput.log("prepare input data ...");
}

template <class Dtype, template <class...> class Qtype>
void OffDataProvider<Dtype, Qtype>::transData() {
  temp_ptrs.clear();
  Dtype* cpuTempData = reinterpret_cast<void**>(cpuTempData_);
  Dtype firstConvTempData = reinterpret_cast<void*>(firstConvData_[0]);
  for (int i = 0; i < this->inBlobNum_; i++) {
    void* temp_ptr = nullptr;
    cnrtTransDataOrder(cpuData_[i],
                      CNRT_FLOAT32,
                      cpuTrans_[i],
                      4,
                      dim_shape,
                      dim_order);
    if (input_data_type[i] != CNRT_FLOAT32) {
      cnrtCastDataType(cpuTrans_[i],
                      CNRT_FLOAT32,
                      (input_data_type[i] == CNRT_UINT8) ? firstConvTempData :
                                                            cpuTempData[i],
                      input_data_type[i],
                      this->input_count,
                      nullptr);

      if (input_data_type[i] == CNRT_UINT8) {
        if (FLAGS_input_format == 0) {
          // input image is rgb(bgr) format
          // input_format=0 represents weight is in rgb(bgr) order
          cnrtAddDataStride(firstConvTempData, CNRT_UINT8, cpuTempData[i], 4,
                            inputDimValue, inputDimStride);
        } else if (FLAGS_input_format == 1 || FLAGS_input_format == 3) {
          // input data is in rgba(bgra) format
          // input_format=1 represents weight is in argb(abgr) order
          uint8_t* tmp = new uint8_t[this->input_count];
          tmp[0] = 0;
          memcpy(tmp + 1, firstConvTempData, this->input_count - 1);
          memcpy(cpuTempData[i], tmp, this->input_count);
          delete [] tmp;
        } else if (FLAGS_input_format == 2) {
          // input data is in bgra(rgba) format
          // input_format=2 represents weight is in bgra(rgba) order
          memcpy(cpuTempData[i], firstConvTempData, this->input_count);
        }
      }
      temp_ptr = cpuTempData[i];
    } else {
      temp_ptr = cpuTrans_[i];
    }
    temp_ptrs.push_back(temp_ptr);
  }
}

template <class Dtype, template <class...> class Qtype>
void OffDataProvider<Dtype, Qtype>::runParallel() {
  OffRunner<Dtype, Qtype> *off_runner = static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);
  init();
  if (FLAGS_perf_mode) {
    int leftNumBatch = FLAGS_perf_mode_img_num / this->inNum_;
    Timer preprocessor;
    readInputData();
    prepareInput();
    transData();
    while (leftNumBatch--) {
      std::shared_ptr<Dtype> mluData = off_runner->popFreeInputData();
      // this queue is for async copy
      cnrtQueue_t queue;
      if (FLAGS_async_copy) {
        queue = off_runner->popFreeAsyncCopyQueue();
      }
      Timer copyin;
      TimePoint t1 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < this->inBlobNum_; i++) {
        if (FLAGS_async_copy) {
          cnrtMemcpyAsync(mluData.get()[i],
                          temp_ptrs[i],
                          off_runner->inputSizeS[i],
                          queue,
                          CNRT_MEM_TRANS_DIR_HOST2DEV);
        } else {
          cnrtMemcpy(mluData.get()[i],
                    temp_ptrs[i],
                    off_runner->inputSizeS[i],
                    CNRT_MEM_TRANS_DIR_HOST2DEV);
        }
      }
      copyin.log("copyin time ...");
      TimePoint t2 = std::chrono::high_resolution_clock::now();
      auto timetrace = off_runner->popFreeInputTimeTraceData();
      timetrace->in_start = t1;
      timetrace->in_end = t2;
      off_runner->pushValidInputTimeTraceData(timetrace);

      if (FLAGS_async_copy) {
        off_runner->pushValidInputDataAndNames(mluData, this->imageNameVec, queue);
      } else {
        off_runner->pushValidInputDataAndNames(mluData, this->imageNameVec);
      }
      preprocessor.log("preprocessor time ...");
    }
  } else {
    while (this->imageList.size()) {
      Timer preprocessor;
      readInputData();
      prepareInput();
      std::shared_ptr<Dtype> mluData = off_runner->popFreeInputData();
      // this queue is for async copy
      cnrtQueue_t queue;
      if (FLAGS_async_copy) {
        queue = off_runner->popFreeAsyncCopyQueue();
      }
      Timer copyin;
      TimePoint t1 = std::chrono::high_resolution_clock::now();
      transData();
      if (temp_ptrs.size() == 0) break;
      for (int i = 0; i < this->inBlobNum_; i++) {
        if (FLAGS_async_copy) {
          cnrtMemcpyAsync(mluData.get()[i],
                          temp_ptrs[i],
                          off_runner->inputSizeS[i],
                          queue,
                          CNRT_MEM_TRANS_DIR_HOST2DEV);
        } else {
          cnrtMemcpy(mluData.get()[i],
                    temp_ptrs[i],
                    off_runner->inputSizeS[i],
                    CNRT_MEM_TRANS_DIR_HOST2DEV);
        }
      }
      copyin.log("copyin time ...");
      TimePoint t2 = std::chrono::high_resolution_clock::now();
      auto timetrace = off_runner->popFreeInputTimeTraceData();
      timetrace->in_start = t1;
      timetrace->in_end = t2;
      off_runner->pushValidInputTimeTraceData(timetrace);

      if (FLAGS_async_copy) {
        off_runner->pushValidInputDataAndNames(mluData, this->imageNameVec, queue);
      } else {
        off_runner->pushValidInputDataAndNames(mluData, this->imageNameVec);
      }
      preprocessor.log("preprocessor time ...");
    }
  }

  LOG(INFO) << "DataProvider: no data ...";
  // tell runner there is no more images
}

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

INSTANTIATE_OFF_CLASS(OffDataProvider);

#endif  // USE_MLU && USE_OPENCV
