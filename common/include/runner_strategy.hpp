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

#ifndef CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_RUNNER_STRATEGY_HPP_
#define CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_RUNNER_STRATEGY_HPP_

#include<vector>
#include<string>
#ifdef USE_MLU
#endif

template <class Dtype, template <class...> class Qtype>
class OffRunner;

template <class Dtype, template <class...> class Qtype>
class RunnerStrategy {
  public:
    virtual void runParallel(OffRunner<Dtype, Qtype>* runner) {}
    virtual ~RunnerStrategy() {}
};

template <class Dtype, template <class...> class Qtype>
class SimpleStrategy: public RunnerStrategy<Dtype, Qtype> {
  public:
    virtual void runParallel(OffRunner<Dtype, Qtype>* runner);
};

template <class Dtype, template <class...> class Qtype>
class FlexibleStrategy: public RunnerStrategy<Dtype, Qtype> {
  public:
    virtual void runParallel(OffRunner<Dtype, Qtype>* runner);
};

#endif  // CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_RUNNER_STRATEGY_HPP_
