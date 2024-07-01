/*
All modification made by Cambricon Corporation: © 2018 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2018, the respective contributors
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_RUNNER_HPP_
#define CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_RUNNER_HPP_
#include <vector>
#include <string>
#include <queue>
#include <sstream>
#include <thread> // NOLINT
#include <utility>
#include <iomanip>
#include <map>
#include <fstream>
#include <mutex> // NOLINT
#ifdef USE_MLU
#include "cnrt.h" // NOLINT
#endif
#include "blocking_queue.hpp"
#include "queue.hpp"
#include "post_processor.hpp"
#include "common_functions.hpp"

// smart pointer to void** for cnrtMalloc
template<typename T>
inline std::shared_ptr<T> MakeArray(int size) {
    return std::shared_ptr<T>( new T[size],
        [size](T *p) {
            for (int i = 0; i < size; i++)
                cnrtFree(p[i]);
            delete[] p;
        });
}

template <class Dtype, template <class...> class Qtype>
class Runner {
  public:
  Runner():initSerialMode(false), simple_flag_(true) {}
  virtual ~Runner() {}
  int n() {return inNum_;}
  int c() {return inChannel_;}
  int h() {return inHeight_;}
  int w() {return inWidth_;}

  void pushValidInputData(std::shared_ptr<Dtype> data) { validInputFifo_.push(data); }
  void pushFreeInputData(std::shared_ptr<Dtype> data) { freeInputFifo_.push(data); }
  std::shared_ptr<Dtype> popValidInputData() { return validInputFifo_.pop(); }
  std::shared_ptr<Dtype> popFreeInputData() { return freeInputFifo_.pop(); }
  void pushValidAsyncCopyQueue(cnrtQueue_t queue) { validAsyncCopyQueueFifo_.push(queue); }
  void pushFreeAsyncCopyQueue(cnrtQueue_t queue) { freeAsyncCopyQueueFifo_.push(queue); }
  cnrtQueue_t popValidAsyncCopyQueue() { return validAsyncCopyQueueFifo_.pop(); }
  cnrtQueue_t popFreeAsyncCopyQueue() { return freeAsyncCopyQueueFifo_.pop(); }
  void pushValidOutputData(std::shared_ptr<Dtype> data) { validOutputFifo_.push(data); }
  void pushFreeOutputData(std::shared_ptr<Dtype> data) { freeOutputFifo_.push(data);}
  std::shared_ptr<Dtype> popValidOutputData() { return validOutputFifo_.pop(); }
  std::shared_ptr<Dtype> popFreeOutputData() { return freeOutputFifo_.pop(); }
  void pushValidInputTempData(Dtype* data) { validInputTempFifo_.push(data);}
  void pushFreeInputTempData(Dtype* data) { freeInputTempFifo_.push(data); }
  Dtype* popValidInputTempData() { return validInputTempFifo_.pop(); }
  Dtype* popFreeInputTempData() { return freeInputTempFifo_.pop(); }
  void pushValidOutputTempData(Dtype* data) { validOutputTempFifo_.push(data); }
  void pushFreeOutputTempData(Dtype* data) { freeOutputTempFifo_.push(data); }
  Dtype* popValidOutputTempData() { return validOutputTempFifo_.pop(); }
  Dtype* popFreeOutputTempData() { return freeOutputTempFifo_.pop(); }
  void pushFreeInputTimeTraceData(std::shared_ptr<InferenceTimeTrace> data) {
    freeInputTimetraceFifo_.push(data);
  }
  void pushValidInputTimeTraceData(std::shared_ptr<InferenceTimeTrace> data) {
    validInputTimetraceFifo_.push(data);
  }
  void pushValidOutputTimeTraceData(std::shared_ptr<InferenceTimeTrace> data) {
    validOutputTimetraceFifo_.push(data);
  }
  std::shared_ptr<InferenceTimeTrace> popValidInputTimeTraceData() {
    return validInputTimetraceFifo_.pop();
  }
  std::shared_ptr<InferenceTimeTrace> popValidOutputTimeTraceData() {
    return validOutputTimetraceFifo_.pop();
  }
  std::shared_ptr<InferenceTimeTrace> popFreeInputTimeTraceData() {
    return freeInputTimetraceFifo_.pop();
  }

  void pushValidInputNames(std::vector<std::pair<std::string, cv::Mat>> images) {
    imagesFifo_.push(images);
  }
  std::vector<std::pair<std::string, cv::Mat>> popValidInputNames() { return imagesFifo_.pop(); }
  void pushValidInputDataAndNames(std::shared_ptr<Dtype> data,
                                  const std::vector<std::pair<std::string, cv::Mat>>& names) {
    std::lock_guard<std::mutex> lk(runner_mutex_);
    pushValidInputData(data);
    pushValidInputNames(names);
  }

  void pushValidInputDataAndNames(std::shared_ptr<Dtype> data,
                                  const std::vector<std::pair<std::string, cv::Mat>>& names,
                                  cnrtQueue_t queue) {
    std::lock_guard<std::mutex> lk(runner_mutex_);
    pushValidInputData(data);
    pushValidInputNames(names);
    pushValidAsyncCopyQueue(queue);
  }

  virtual void runParallel() {}

  inline int inBlobNum() { return inBlobNum_; }
  inline int inCount() { return inCount_; }
  inline int outBlobNum() { return outBlobNum_; }
  inline int outCount() { return outCount_; }
  inline int outNum() { return outNum_; }
  inline int outChannel() { return outChannel_; }
  inline int outHeight() { return outHeight_; }
  inline int outWidth() { return outWidth_; }
  inline int threadId() { return threadId_; }
  inline int deviceId() { return deviceId_; }
  inline int deviceSize() { return deviceSize_; }
  inline float runTime() { return runTime_; }
  inline void setRunTime(const float& time) {runTime_ = time;}
  inline void setThreadId(int id) { threadId_ = id; }
  inline void setPostProcessor(PostProcessor<Dtype, Qtype> *p ) { postProcessor_ = p; }
  inline std::vector<int> outAllCounts() { return outallcounts_; }
  inline bool simpleFlag() {return simple_flag_;}

  private:
  Qtype<std::shared_ptr<Dtype>> validInputFifo_;
  Qtype<std::shared_ptr<Dtype>> freeInputFifo_;
  Qtype<cnrtQueue_t> validAsyncCopyQueueFifo_;
  Qtype<cnrtQueue_t> freeAsyncCopyQueueFifo_;
  Qtype<std::shared_ptr<Dtype>> validOutputFifo_;
  Qtype<std::shared_ptr<Dtype>> freeOutputFifo_;
  Qtype<Dtype*> validInputTempFifo_;
  Qtype<Dtype*> freeInputTempFifo_;
  Qtype<Dtype*> validOutputTempFifo_;
  Qtype<Dtype*> freeOutputTempFifo_;
  Qtype<std::vector<std::pair<std::string, cv::Mat>>> imagesFifo_;
  Qtype<std::shared_ptr<InferenceTimeTrace>> freeInputTimetraceFifo_;
  Qtype<std::shared_ptr<InferenceTimeTrace>> validInputTimetraceFifo_;
  Qtype<std::shared_ptr<InferenceTimeTrace>> validOutputTimetraceFifo_;
  std::mutex runner_mutex_;

  protected:
  int inBlobNum_, outBlobNum_;
  unsigned int inNum_, inChannel_, inHeight_, inWidth_;
  unsigned int outNum_, outChannel_, outHeight_, outWidth_;
  std::vector<int> outallcounts_;
  int inCount_, outCount_;
  int threadId_;
  int deviceId_;
  int deviceSize_ = 1;
  int Parallel_ = 1;
  float runTime_;
  bool initSerialMode;
  bool simple_flag_;

  PostProcessor<Dtype, Qtype> *postProcessor_;
};
#endif  // CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_RUNNER_HPP_
