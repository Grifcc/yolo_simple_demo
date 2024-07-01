/*
All modification made by Cambricon Corporation: © 2019 Cambricon Corporation
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

#ifndef EXAMPLES_OFFLINE_COMMON_INCLUDE_THREADPOOL_H_
#define EXAMPLES_OFFLINE_COMMON_INCLUDE_THREADPOOL_H_
#include <thread>  // NOLINT
#include <mutex>   // NOLINT
#include <atomic>
#include <condition_variable>  // NOLINT
#include <functional>
#include <future>  // NOLINT
#include <queue>
#include <type_traits>
#include <utility>
#include <vector>
#include "glog/logging.h"

namespace zl {
class ThreadsGuard {
  public:
    explicit ThreadsGuard(std::vector<std::thread>& v) : threads_(v) {}
    ~ThreadsGuard() {
      for (size_t i = 0; i != threads_.size(); ++i) {
        if (threads_[i].joinable()) {
          threads_[i].join();
        }
      }
    }
  private:
    ThreadsGuard(ThreadsGuard&& tg) = delete;
    ThreadsGuard& operator = (ThreadsGuard&& tg) = delete;
    ThreadsGuard(const ThreadsGuard&) = delete;
    ThreadsGuard& operator = (const ThreadsGuard&) = delete;

  private:
    std::vector<std::thread>& threads_;
};

class ThreadPool {
  public:
    typedef std::function<void()> task_type;

  public:
    explicit ThreadPool(int n = 0);

    ~ThreadPool() {
      stop();
      cond_.notify_all();
      for (size_t i = 0; i != threads_.size(); ++i) {
        if (threads_[i].joinable()) {
          threads_[i].join();
        }
      }

    }

    void stop() {
      stop_.store(true, std::memory_order_release);
    }

    template<class Function, class... Args>
    // std::future<typename std::result_of<Function(Args...)>::type>
    void add(Function&&, Args&&...);

  private:
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator = (ThreadPool&&) = delete;
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator = (const ThreadPool&) = delete;

  private:
    std::atomic<bool> stop_;
    std::mutex mtx_;
    std::condition_variable cond_;
    std::queue<task_type> tasks_;
    std::vector<std::thread> threads_;
    zl::ThreadsGuard tg_;
    std::mutex push_mtx_;
    std::condition_variable push_cond_;
};

inline ThreadPool::ThreadPool(int n) : stop_(false), tg_(threads_) {
  int nthreads = n;
  if (nthreads <= 0) {
    nthreads = std::thread::hardware_concurrency();
    nthreads = (nthreads == 0 ? 2 : nthreads);
  }

  for (int i = 0; i != nthreads; ++i) {
    threads_.push_back(std::thread([this]{
      while (!stop_.load(std::memory_order_acquire) || !this->tasks_.empty()) {
        task_type task;
        {
          std::unique_lock<std::mutex> ulk(this->mtx_);
          this->cond_.wait(ulk, [this]{ return stop_.load(
                std::memory_order_acquire) ||!this->tasks_.empty(); });
          if (stop_.load(std::memory_order_acquire) && this->tasks_.empty()) {
            return;
          }
          task = std::move(this->tasks_.front());
          this->tasks_.pop();
          this->push_cond_.notify_one();
        }
        task();
      }
    }));
  }
}

template<class Function, class... Args>
void ThreadPool::add(Function&& fcn, Args&&... args) {
  auto fn = std::bind(std::forward<Function>(fcn), std::forward<Args>(args)...);
  {
    std::lock_guard<std::mutex> lg(mtx_);
    if (stop_.load(std::memory_order_acquire))
      throw std::runtime_error("thread pool has stopped");
  }
  {
    std::unique_lock<std::mutex> ulk(this->push_mtx_);
    push_cond_.wait(ulk, [this]{ return this->tasks_.size() <= 2;});
  }
  {
    std::lock_guard<std::mutex> lg(mtx_);
    tasks_.push(fn);
  }
  cond_.notify_one();
  return;
}
}  // namespace zl

#endif  // EXAMPLES_OFFLINE_COMMON_INCLUDE_THREADPOOL_H_
