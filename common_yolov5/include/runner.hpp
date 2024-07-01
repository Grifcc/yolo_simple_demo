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
#include "handler.hpp"

class Runner {
  public:
  Runner() {}
  Runner(OfflineDescripter* offline_descripter,
            const int& id);
  virtual ~Runner();

  virtual void runParallel();

  inline int threadId() { return threadId_; }
  inline int deviceId() { return deviceId_; }
  inline float runTime() { return runTime_; }
  inline void setRunTime(const float& time) {runTime_ = time;}
  inline void setThreadId(int id) { threadId_ = id; }

  private:
  std::mutex runner_mutex_;

  protected:
  int threadId_;
  int deviceId_;
  float runTime_;

  public:
  OfflineDescripter* offlineDescripter() { return offline_descripter_; }

  private:
  // TODO: czr - use shared_ptr instead of raw pointer
  OfflineDescripter* offline_descripter_;
};
#endif  // CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_RUNNER_HPP_
