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

#ifndef CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_OFF_RUNNER_HPP_
#define CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_OFF_RUNNER_HPP_
#if defined(USE_MLU)
#include <vector>
#include <string>
#include "cnrt.h" // NOLINT
#include "blocking_queue.hpp"
#include "queue.hpp"
#include "post_processor.hpp"
#include "runner.hpp"

template <class Dtype, template <class...> class Qtype>
class RunnerStrategy;

template <class Dtype, template <class...> class Qtype>
class OffRunner : public Runner<Dtype, Qtype>{
  public:
  OffRunner(const std::string& offlinemodel,
            const int& id,
            const int& parallel,
            const int& deviceId,
            const int& devicesize);
  OffRunner(const cnrtRuntimeContext_t rt_ctx,
            const int& id);
  virtual ~OffRunner();

  virtual void runParallel();
  cnrtFunction_t function() { return function_; }
  cnrtQueue_t queue() { return queue_; }
  void setQueue(const cnrtQueue_t& queue) { queue_ = queue; }
  cnrtRuntimeContext_t runtimeContext() { return rt_ctx_; }
  int64_t *inputSizeS, *outputSizeS;

  private:
  void getIODataDesc();
  void loadOfflinemodel(const std::string& offlinemodel);

  private:
  RunnerStrategy<Dtype, Qtype>* runnerStrategy_;
  cnrtModel_t model_;
  cnrtQueue_t queue_;
  cnrtFunction_t func;
  cnrtFunction_t function_;
  cnrtRuntimeContext_t rt_ctx_;
  cnrtDim3_t dim_;
};
#endif  // USE_MLU
#endif  // CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_OFF_RUNNER_HPP_
