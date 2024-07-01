#ifndef CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_HANDLER_HPP_
#define CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_HANDLER_HPP_
#include <vector>
#include <string>
#include <thread> // NOLINT
#include <memory>
#include <algorithm>

#ifdef USE_MLU
#include "cnrt.h" // NOLINT
#endif
#include "common_functions.hpp"
#include "blocking_queue.hpp"
#include "queue.hpp"

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

// smart pointer for custom deleter
template<typename T, typename Deleter>
inline std::shared_ptr<T> MakeArray(int size, Deleter d) {
    return std::shared_ptr<T>( new T[size],
        [size, d](T *p) {
            for (int i = 0; i < size; i++)
                d(p[i]);
            delete[] p;
        });
}

class BoxingData {
public:
  void setQueue(std::shared_ptr<cnrtQueue_t> queue) {
    if (FLAGS_async_copy) {
      if (queue == nullptr) {
        CHECK(cnrtCreateQueue(queue.get()) == CNRT_RET_SUCCESS)
          << "CNRT create queue for async copy failed";
      }
      queue_ = queue;
    } else {
      queue_ = nullptr;
    }
  }

  std::shared_ptr<cnrtQueue_t> getQueue() {
    return queue_;
  }

  void setBuf(std::shared_ptr<void*> buffer) {
    buffer_ = buffer;
  }

  std::shared_ptr<void*> getBuf() {
    return buffer_;
  }

  void setStamp(std::shared_ptr<InferenceTimeTrace> timestamp) {
    stamp_ = timestamp;
  }

  std::shared_ptr<InferenceTimeTrace> getStamp() {
    return stamp_;
  }

  void setImageAndName(const std::vector<std::pair<std::string, cv::Mat>>& image_and_name) {
    image_and_name_ = std::move(image_and_name);
  }

  std::vector<std::pair<std::string, cv::Mat>> getImageAndName() {
    // TODO: maybe shared pointer is better to avoid copy construct
    return image_and_name_;
  }

  void setFeatureName(std::shared_ptr<std::vector<InputFeatures>> feat_files) {
    // feat_files_ = std::move(feat_files);
    feat_files_ = feat_files;
  }

  std::shared_ptr<std::vector<InputFeatures>> getFeatureName() {
    return feat_files_;
  }

  // TODO: buffer stores mlu data, maybe mlu_buffer_ is more proper
  std::shared_ptr<void*> buffer_;
  std::shared_ptr<InferenceTimeTrace> stamp_;
  std::shared_ptr<cnrtQueue_t> queue_;
  std::vector<std::pair<std::string, cv::Mat>> image_and_name_;
  std::shared_ptr<std::vector<InputFeatures>> feat_files_;
};

class ShapeDescripter {
public:
  // TODO: czr - initialization
  /* explicit ShapeDescripter(unsigned inputNum, unsigned outputNum) : inputNum_(inputNum), outputNum_(outputNum) { */
  /* } */

  void setShape(unsigned inputNum, unsigned outputNum) {
    inputNum_ = inputNum;
    outputNum_ = outputNum;
    // XXX: resize here to make sure indexing afterwards
    inputShapes_.resize(inputNum);
    outputShapes_.resize(outputNum);
  }

  void setOutputShapes(std::shared_ptr<int*> shape, unsigned index) {
    if (index >= outputShapes_.size()) LOG(ERROR) << "illegal index in setting output shapes";
    outputShapes_[index].push_back(shape.get()[0][0]); // set n
    outputShapes_[index].push_back(shape.get()[0][3]); // set c
    outputShapes_[index].push_back(shape.get()[0][1]); // set h
    outputShapes_[index].push_back(shape.get()[0][2]); // set w
  }

  inline std::vector<unsigned> getOutputShape(unsigned index = 0) {
    return outputShapes_[index];
  }

  inline std::vector<std::vector<unsigned>> getOutputShapeS() {
    return outputShapes_;
  }
  inline unsigned outputNum() {
    return outputNum_;
  }

  inline unsigned outCount(int index = 0) {
    if (index >= outputShapes_.size()) LOG(ERROR) << "illegal index in getting output shapes";
    unsigned counts = 1;
    auto retrieved_shape = outputShapes_[index];
    for (auto channel_shape :retrieved_shape) {
      counts *= channel_shape;
    }
    return counts;
  }

  void setInputShapes(std::shared_ptr<int*> shape, unsigned index, bool is_first_conv) {
    if (index >= inputShapes_.size()) LOG(ERROR) << "illegal index in setting input shapes";
    inputShapes_[index].push_back(shape.get()[0][0]); // set n
    auto channel = shape.get()[0][3];
    // input channel is set to 4 if is first conv.
    // input channel needs -- if the first input is firstconv and use rgb input_format
    if (is_first_conv && (index == 0) && (FLAGS_input_format == 0)) {
      --channel;
    }
    inputShapes_[index].push_back(channel); // set c
    inputShapes_[index].push_back(shape.get()[0][1]); // set h
    inputShapes_[index].push_back(shape.get()[0][2]); // set w
  }

  inline std::vector<unsigned> getInputShape(unsigned index = 0) {
    return inputShapes_[index];
  }

  inline unsigned inputNum() {
    return inputNum_;
  }

  inline unsigned inCount(int index = 0) {
    if (index >= inputShapes_.size()) LOG(ERROR) << "illegal index in getting input shapes";
    unsigned counts = 1;
    for (auto channel_shape : inputShapes_[index]) {
      counts *= channel_shape;
    }
    return counts;
  }

  inline unsigned outN(int index = 0) {
    if (index >= outputShapes_.size()) LOG(ERROR) << "illegal index in getting output shapes";
    return outputShapes_[index][0];
  }

  inline unsigned outC(int index = 0) {
    if (index >= outputShapes_.size()) LOG(ERROR) << "illegal index in getting output shapes";
    return outputShapes_[index][1];
  }

  inline unsigned outH(int index = 0) {
    if (index >= outputShapes_.size()) LOG(ERROR) << "illegal index in getting output shapes";
    return outputShapes_[index][2];
  }

  inline unsigned outW(int index = 0) {
    if (index >= outputShapes_.size()) LOG(ERROR) << "illegal index in getting output shapes";
    return outputShapes_[index][3];
  }

  inline unsigned inN(int index = 0) {
    if (index >= inputShapes_.size()) LOG(ERROR) << "illegal index in getting input shapes";
    return inputShapes_[index][0];
  }

  inline unsigned inC(int index = 0) {
    if (index >= inputShapes_.size()) LOG(ERROR) << "illegal index in getting input shapes";
    return inputShapes_[index][1];
  }

  inline unsigned inH(int index = 0) {
    if (index >= inputShapes_.size()) LOG(ERROR) << "illegal index in getting input shapes";
    return inputShapes_[index][2];
  }

  inline unsigned inW(int index = 0) {
    if (index >= inputShapes_.size()) LOG(ERROR) << "illegal index in getting input shapes";
    return inputShapes_[index][3];
  }

private:
  unsigned inputNum_;
  unsigned outputNum_;
  std::vector<std::vector<unsigned>> inputShapes_;
  std::vector<std::vector<unsigned>> outputShapes_;
};

class OfflineDescripter {

public:
  OfflineDescripter() {
    shape_descripter_ = std::make_shared<ShapeDescripter>();
  }
  OfflineDescripter(const std::vector<int>& deviceIds) : deviceIds_(deviceIds) {
    shape_descripter_ = std::make_shared<ShapeDescripter>();
    int size = *max_element(deviceIds.begin(), deviceIds.end()) +1; 
    validInputBoxingFifo_.resize(size);
    freeInputBoxingFifo_.resize(size);
    validOutputBoxingFifo_.resize(size);
    freeOutputBoxingFifo_.resize(size);
  }
  ~OfflineDescripter() {
    cnrtDestroyFunction(function_);
  }


  // ShapeDescripter related functions
  inline unsigned outputNum() { return shape_descripter_->outputNum(); }
  inline unsigned inputNum() { return shape_descripter_->inputNum(); }
  inline unsigned inCount(int index = 0) { return shape_descripter_->inCount(index); }
  inline unsigned outCount(int index = 0) { return shape_descripter_->outCount(index); }
  inline unsigned outN(int index = 0) { return shape_descripter_->outN(index); }
  inline unsigned outC(int index = 0) { return shape_descripter_->outC(index); }
  inline unsigned outH(int index = 0) { return shape_descripter_->outH(index); }
  inline unsigned outW(int index = 0) { return shape_descripter_->outW(index); }
  inline unsigned inN(int index = 0) { return shape_descripter_->inN(index); }
  inline unsigned inC(int index = 0) { return shape_descripter_->inC(index); }
  inline unsigned inH(int index = 0) { return shape_descripter_->inH(index); }
  inline unsigned inW(int index = 0) { return shape_descripter_->inW(index); }

  inline std::vector<unsigned> getInputShape(unsigned index = 0) const {
    return shape_descripter_->getInputShape(index);
  }

  inline std::vector<unsigned> getOutputShape(unsigned index = 0) const {
    return shape_descripter_->getOutputShape(index);
  }

  inline std::vector<std::vector<unsigned>> getOutputShapeS() const {
    return shape_descripter_->getOutputShapeS();
  }
  // data type related functions
  cnrtDataType_t* getInputDataType() const {
    return input_data_type_;
  }
  cnrtDataType_t* getOutputDataType() const {
    return output_data_type_;
  }

  inline bool isFirstConv() const { return is_first_conv_; }

  // BoxingData push/pop
  void pushFreeInputBoxingData(std::shared_ptr<BoxingData> boxing_data, int id) {freeInputBoxingFifo_[id].push(boxing_data);};
  void pushFreeOutputBoxingData(std::shared_ptr<BoxingData> boxing_data, int id) {freeOutputBoxingFifo_[id].push(boxing_data);};
  void pushValidInputBoxingData(std::shared_ptr<BoxingData> boxing_data, int id) { validInputBoxingFifo_[id].push(boxing_data); }
  void pushValidOutputBoxingData(std::shared_ptr<BoxingData> boxing_data, int id) { validOutputBoxingFifo_[id].push(boxing_data); }
  std::shared_ptr<BoxingData> popValidInputBoxingData(int id) { return validInputBoxingFifo_[id].pop(); }
  std::shared_ptr<BoxingData> popFreeInputBoxingData(int id) { return freeInputBoxingFifo_[id].pop(); }
  std::shared_ptr<BoxingData> popValidOutputBoxingData(int id) { return validOutputBoxingFifo_[id].pop(); }
  std::shared_ptr<BoxingData> popFreeOutputBoxingData(int id) { return freeOutputBoxingFifo_[id].pop(); }

  // TODO: Following are from runner
  cnrtFunction_t function() { return function_; }
  cnrtQueue_t queue() { return queue_; }
  void setQueue(const cnrtQueue_t& queue) { queue_ = queue; }
  cnrtRuntimeContext_t runtimeContext() { return rt_ctx_; }

private:
  cnrtFunction_t function_;
  cnrtQueue_t queue_;
  cnrtRuntimeContext_t rt_ctx_;
  cnrtDataType_t* output_data_type_ = nullptr;
  cnrtDataType_t* input_data_type_ = nullptr;

public:
  // TODO: temporarily use raw pointer and make it public
  // it's better use shared_ptr and private value
  int64_t *inputSizeS, *outputSizeS;

  std::vector<BlockingQueue<std::shared_ptr<BoxingData>>> validInputBoxingFifo_;
  std::vector<BlockingQueue<std::shared_ptr<BoxingData>>> freeInputBoxingFifo_;
  std::vector<BlockingQueue<std::shared_ptr<BoxingData>>> validOutputBoxingFifo_;
  std::vector<BlockingQueue<std::shared_ptr<BoxingData>>> freeOutputBoxingFifo_;

  // TODO: Following are from simple interface
public:
  void loadOfflinemodel(const std::string& offlinemodel,
                        const bool& channel_dup,
                        const int threads);
  std::vector<std::vector<cnrtRuntimeContext_t>> getRuntimeContexts() {
    return dev_runtime_contexts_;
  }
  cnrtRuntimeContext_t getRuntimeContext(const int id) {
    // Given D devices and T threads. Runner uses Ith thread (I<T).
    // rtx_ used in Runner is stored in dev_runtime_contexts_[I%D][I/D]
    return dev_runtime_contexts_[id % deviceIds_.size()][id / deviceIds_.size()];
  }
  void destroyRuntimeContext();
  // 4 threads of data providers and 2 threads of post processors
  static const int data_provider_num_ = 4;
  static const int postProcessor_num_ = 2;
  // 1 threads of thread pool
  static const int thread_num = 1;
private:
  void prepareRuntimeContext(cnrtRuntimeContext_t *ctx, cnrtFunction_t function,
                             int deviceId, const bool& channel_dup);
private:
  std::vector<int> deviceIds_;
  cnrtModel_t model_;
  std::vector<std::vector<cnrtRuntimeContext_t>> dev_runtime_contexts_;

  std::shared_ptr<ShapeDescripter> shape_descripter_;
  bool is_first_conv_;

  void getIODataDesc();
  void allocateMemory(int queueLength, int deviceId);
};

#endif  // CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_HANDLER_HPP_
