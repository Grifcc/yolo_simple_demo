#ifndef CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_DATA_PROVIDER_HPP_
#define CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_DATA_PROVIDER_HPP_
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
#include "runner.hpp"

template<class Dtype, template <class...> class Qtype>
class DataProvider {
  public:
  explicit DataProvider(const std::string& meanvalue,
                        const std::string& stdvalue,
                        const std::queue<std::string>& images):
                        threadId_(0), deviceId_(0),
                        meanValue_(meanvalue),
                        stdValue_(stdvalue),
                        imageList(images), initSerialMode(false) {}
  virtual ~DataProvider() {
  }
  void readOneBatch();
  bool imageIsEmpty();
  void preRead();
  void SetMean();
  void SetStd();
  virtual void SetMeanValue();
  virtual void SetStdValue();
  void WrapInputLayer(std::vector<std::vector<cv::Mat>>* wrappedImages, float* inputData);
  virtual void resizeMat(const cv::Mat& sample, cv::Mat& sample_resized);
  virtual void normalizeMat(const cv::Mat& sample_float, cv::Mat& sample_normalized);
  virtual void Preprocess(const std::vector<std::pair<std::string, cv::Mat>>& srcImages,
                          std::vector<std::vector<cv::Mat> >* dstImages);
  inline void setThreadId(int id) { threadId_ = id; }
  inline void setRunner(Runner<Dtype, Qtype> *p) {
    runner_ = p;
    inNum_ = p->n();  // make preRead happy
  }
  virtual void runParallel() {}

  protected:
  int inBlobNum_;
  int inNum_;
  int inChannel_;
  int inHeight_;
  int inWidth_;
  cv::Size inGeometry_;

  int threadId_;
  int deviceId_;

  std::string meanValue_;
  cv::Mat mean_;

  std::string stdValue_;
  cv::Mat std_;

  std::queue<std::string> imageList;
  std::vector<std::vector<std::pair<std::string, cv::Mat>>> inImageAndName_;
  std::vector<std::pair<std::string, cv::Mat>> imageNameVec;

  bool initSerialMode;

  Runner<Dtype, Qtype> *runner_;
};
#endif  //  USE_OPENCV
#endif  //  CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_DATA_PROVIDER_HPP_
