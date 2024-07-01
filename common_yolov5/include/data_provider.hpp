#ifndef CATCH_EXAMPLES_INFERENCE_COMMON_INCLUDE_DATA_PROVIDER_HPP_
#define CATCH_EXAMPLES_INFERENCE_COMMON_INCLUDE_DATA_PROVIDER_HPP_
#include <queue>
#include <string>
#include <vector>
#include <sstream>
#include <thread> // NOLINT
#include <utility>
#include <iomanip>
#include <fstream>
#include <map>
#if defined(USE_OPENCV)
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common_functions.hpp"
#include "handler.hpp"

class DataProvider {
  public:
  explicit DataProvider(const std::string& meanvalue,
                        const std::string& stdvalue,
                        const std::queue<std::string>& images):
                        threadId_(0), deviceId_(0),
                        meanValue_(meanvalue),
                        stdValue_(stdvalue),
                        imageList(images) {}
  explicit DataProvider():threadId_(0), deviceId_(0){}

  virtual ~DataProvider() {
    // TODO: memory leak risk here
    int inputNum = this->offlineDescripter()->inputNum();
    for (int i = 0; i < inputNum; ++i){
      delete [] reinterpret_cast<float*>(cpuData_[i]);
      delete [] reinterpret_cast<float*>(cpuTrans_[i]);
      delete [] reinterpret_cast<char*>(cpuCastData_[i]);
    }
    delete cpuData_;    
    delete cpuTrans_;    
    delete cpuCastData_;
  }
  void readOneBatch(std::vector<std::pair<std::string, cv::Mat>>& ImageAndName);
  bool imageIsEmpty();
  void SetMean();
  void SetStd();
  virtual void convertColor(const cv::Mat& sourceImage, cv::Mat& sample);
  virtual void resizeMat(const cv::Mat& sample, cv::Mat& sample_resized);
  virtual void convertFloat(const cv::Mat& sample_resized, cv::Mat& sample_float);
  virtual void normalizeMat(const cv::Mat& sample_float, cv::Mat& sample_normalized);
  virtual void SetMeanValue();
  virtual void SetStdValue();
  void WrapInputLayer(std::vector<std::vector<cv::Mat>>* wrappedImages, float* inputData);
  virtual void Preprocess(const std::vector<std::pair<std::string, cv::Mat>>& srcImages,
                          std::vector<std::vector<cv::Mat> >* dstImages);
  void transData();
  void copyin(std::shared_ptr<BoxingData> input_boxing_data);
  inline void setThreadId(int id) { threadId_ = id; }
  inline void setDeviceId(int id) { deviceId_ = id; }
  inline void setOfflineDescripter(OfflineDescripter* pod) {
      offline_descripter_ = pod;
  }
  virtual void runParallel();
  // TODO: use shared_ptr
  inline OfflineDescripter* offlineDescripter() {
      return offline_descripter_;
  }

  protected:
  cv::Size inGeometry_;

  int threadId_;
  int deviceId_;

  std::string meanValue_;
  cv::Mat mean_;

  std::string stdValue_;
  cv::Mat std_;

  std::queue<std::string> imageList;

//  private:
  void** cpuData_;
  void** cpuTrans_;
  void** cpuCastData_;
  std::shared_ptr<uint8_t> cpuStrideData_;
  OfflineDescripter* offline_descripter_;
  void prepareBuffer();
  std::vector<void*> temp_ptrs;

};
#endif  //  USE_OPENCV
#endif  //  CATCH_EXAMPLES_INFERENCE_COMMON_INCLUDE_DATA_PROVIDER_HPP_
