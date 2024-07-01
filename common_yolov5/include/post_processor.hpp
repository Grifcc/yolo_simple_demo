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

#ifndef CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_POST_PROCESSOR_HPP_
#define CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_POST_PROCESSOR_HPP_
#define GET_OUT_TENSOR_SIZE(outdata_type, outdata_size) \
  (((outdata_type) == CNRT_FLOAT16) ? ((outdata_size) * 2) : ((outdata_size) * 4))

#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <thread> // NOLINT
#include <utility>
#include <queue>
#include <map>
#include <fstream>
#include <mutex>
#include "glog/logging.h"
#include "common_functions.hpp"
#include "handler.hpp"

class PostProcessor {
  public:
  PostProcessor() : threadId_(0), deviceId_(0) {}
  virtual ~PostProcessor() {}
  virtual void runParallel() {}
  inline void setThreadId(int id) { threadId_ = id; }
  inline void setDeviceId(int id) { deviceId_ = id; }
  virtual int top1() { return 0;}
  virtual int top5() { return 0;}
  virtual double getPSNR() { return 0;}
  virtual std::vector<InferenceTimeTrace> timeTraces() { return timeTraces_; }
  void appendTimeTrace(InferenceTimeTrace t) { timeTraces_.push_back(t); }
  inline void setOfflineDescripter(OfflineDescripter* pod) {
    offline_descripter_ = pod;
  }
  inline OfflineDescripter* offlineDescripter() {
    return offline_descripter_;
  }

  protected:
  int threadId_;
  int deviceId_;
  int total_ = 0;
  // TODO: make raw pointer as shared_ptr
  OfflineDescripter* offline_descripter_;

  std::vector<InferenceTimeTrace> timeTraces_;

  static std::mutex post_mutex_;
};

class NormalizedBBox {
  public:
  NormalizedBBox() { has_set_size = false; }
  ~NormalizedBBox() {}

  float xmin() const { return xmin_; }
  float xmax() const { return xmax_; }
  float ymin() const { return ymin_; }
  float ymax() const { return ymax_; }
  float size() const { return size_; }
  float score() const {return score_;}
  bool has_size() const {
    if ( has_set_size )
      return true;
    else
      return false;
  }

  void set_xmin(float value) { xmin_ = value; }
  void set_xmax(float value) { xmax_ = value; }
  void set_ymin(float value) { ymin_ = value; }
  void set_ymax(float value) { ymax_ = value; }
  void set_size(float value) {
    size_ = value;
    has_set_size = true;
  }
  void set_score(float value) { score_ = value; }

  private:
  float xmin_;
  float ymin_;
  float xmax_;
  float ymax_;
  int label_;
  bool difficult_;
  float score_;
  float size_;
  bool has_set_size;
};

typedef std::map<int, std::vector<NormalizedBBox> > LabelBBox;

#endif  // CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_POST_PROCESSOR_HPP_
