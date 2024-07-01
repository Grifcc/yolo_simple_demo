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

#ifndef CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_PIPELINE_HPP_
#define CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_PIPELINE_HPP_
#include <atomic>
#include <condition_variable> // NOLINT
#include <queue>
#include <string>
#include <thread> // NOLINT
#include "data_provider.hpp"
#include "runner.hpp"
#include "post_processor.hpp"
#include "command_option.hpp"
#include "common_functions.hpp"

class Pipeline {
  public:
  Pipeline(DataProvider *provider,
           Runner *runner,
           PostProcessor *postprocessor);
  Pipeline(const std::vector<DataProvider*>& providers,
           Runner *runner,
           PostProcessor *postprocessor);
  Pipeline(const std::vector<DataProvider*>& providers,
           Runner *runner,
           const std::vector<PostProcessor*>& postprocessors);
  ~Pipeline();
  void runParallel();
  inline DataProvider* dataProvider() { return data_provider_; }
  inline std::vector<DataProvider*> dataProviders() { return data_providers_; }
  inline Runner* runner() { return runner_; }
  inline PostProcessor* postProcessor() { return postProcessor_; }
  inline std::vector<PostProcessor*> postProcessors() { return postProcessors_; }
  static void notifyAll();
  static void waitForNotification();

  private:
  DataProvider *data_provider_;
  std::vector<DataProvider*> data_providers_;
  Runner *runner_;
  PostProcessor* postProcessor_;
  std::vector<PostProcessor*> postProcessors_;

  static int imageNum;
  static std::vector<std::queue<std::string>> imageList;

  static std::condition_variable condition;
  static std::mutex condition_m;
  static int start;
  static std::vector<std::thread*> stageThreads;
  static std::vector<Pipeline*> pipelines;
};
#endif  // CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_PIPELINE_HPP_
