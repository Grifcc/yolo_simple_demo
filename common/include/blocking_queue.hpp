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

#ifndef CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_BLOCKING_QUEUE_HPP_
#define CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_BLOCKING_QUEUE_HPP_

#include <vector>
#include <memory>
#include <queue>
#include <string>
#include <map>
#if defined(USE_OPENCV)
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

template<typename T>
class BlockingQueue {
  public:
  BlockingQueue();

  void push(const T& t);

  bool try_pop(T* t);

  // This logs a message if the threads needs to be blocked
  // useful for detecting e.g. when data feeding is too slow
  T pop(const std::string& log_on_wait = "");

  bool try_peek(T* t);

  // Return element without removing it
  T peek();

  size_t size() const;

  protected:
  /**
   Move synchronization fields out instead of including boost/thread.hpp
   to avoid a boost/NVCC issues (#1009, #1010) on OSX. Also fails on
   Linux CUDA 7.0.18.
   */
  class sync;

  std::queue<T> queue_;
  std::shared_ptr<sync> sync_;

// DISABLE_COPY_AND_ASSIGN(BlockQueue);
};

// A single set of features of data.
// Squad batch_size is based on features
struct InputFeatures {
  // A single class of features of SQUADv1.1 data
  int unique_id_;
  int example_index_;
  int doc_span_index_;
  int paragraph_len_;
  std::vector<std::string> tokens_;
  std::map<int, int> token_to_orig_map_;
  std::map<int, bool> token_is_max_context_;
  std::vector<int> input_ids_;         // input 0
  std::vector<int> segment_ids_;       // input 1
  std::vector<short int> input_mask_;  // input 2
  std::vector<int> p_mask_;
};

#endif  // USE_OPENCV
#endif  // CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_BLOCKING_QUEUE_HPP_
