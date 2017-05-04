
/** @file attributes_recognizer.cpp
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cassert>

#include <vector>
#include <string>

#include <DeepMARCaffe2.hpp>
#include <caffe2/core/net.h>
#include <caffe2/core/init.h>
#include <caffe2/core/predictor.h>
#include <caffe2/core/common_gpu.h>
#include <caffe2/utils/proto_utils.h>

using namespace std;
using namespace caffe2;

namespace cripac {

void DeepMAR::setDevice() {
#ifndef CPU_ONLY
  SetDefaultGPUID(gpuIndex);
#endif
}

/**
 * Initialize the DeepMAR network with a protocol buffer file and a model file.
 * @param init_net_path
 * @param predict_net_path
 * @param gpu_index
 * @return 0 on success; negative values on failure.
 */
int DeepMAR::initialize(const char *init_net_path,
                        const char *predict_net_path,
                        int gpu_index) {
  // Check input.
  if (init_net_path == nullptr) {
    fprintf(stderr, "Error: protocol buffer path is nullptr at file %s, line %d\n",
            __FILE__, __LINE__);
    fflush(stdout), fflush(stderr);
    return DEEPMAR_ILLEGAL_ARG;
  }
  // predict_net_path input.
  if (predict_net_path == nullptr) {
    fprintf(stderr, "Error: Caffe model path is nullptr at file %s, line %d\n",
            __FILE__, __LINE__);
    fflush(stdout), fflush(stderr);
    return DEEPMAR_ILLEGAL_ARG;
  }

  gpuIndex = gpu_index;
  // Set device so that the memory can be correctly allocated.
  setDevice();
  fprintf(stdout, "Using device %d.\n", gpuIndex);
  fflush(stdout), fflush(stderr);
  
  // Load the network.
  fprintf(stdout, "Loading protocol from %s...\n", init_net_path);
  fflush(stdout), fflush(stderr);
  NetDef init_net, predict_net;
  CAFFE_ENFORCE(ReadProtoFromFile(init_net_path, &init_net));
  CAFFE_ENFORCE(ReadProtoFromFile(predict_net_path, &predict_net));
  VLOG(1) << "Init net: " << ProtoDebugString(init_net);
  LOG(INFO) << "Predict net: " << ProtoDebugString(predict_net);

  // Create predictor.
  predictor_ = make_unique<Predictor>(init_net, predict_net);

  fflush(stdout), fflush(stderr);
  return DEEPMAR_OK;
}

DeepMAR::DeepMAR() {
  input_tensors_.push_back(new TensorCPU());
}

DeepMAR::~DeepMAR() {
  delete input_tensors_[0];
}

// Fetch the data of fc8.
const float* DeepMAR::recognize(const float *input) {
  assert(input != nullptr);

  // In case this instance is used in another thread.
  setDevice();

  // Put the input into input blob.
  if (currentBatchSize != 1)
    input_tensors_[0]->Resize(currentBatchSize = 1, 3, kInputHeight, kInputWidth);
  memcpy(input_tensors_[0]->mutable_data<float>(), input, sizeof(float) * kInputHeight * kInputWidth * 3);

  predictor_->run(input_tensors_, &output_tensors_);

  auto *fc8 = predictor_->ws()->GetBlob("fc8");
  CAFFE_ENFORCE(fc8, "Blob: fc8 does not exist");
  return fc8->template Get<TensorCPU>().data<float>();
}

const float *DeepMAR::recognize(int numImages, float *data[]) {
  assert(data != nullptr);

  // In case this instance is used in another thread.
  setDevice();

  if (currentBatchSize != numImages)
    input_tensors_[0]->Resize(currentBatchSize = numImages, 3, kInputHeight, kInputWidth);

  float *dst = input_tensors_[0]->mutable_data<float>();
  for (int i = 0; i < numImages; ++i) {
    memcpy(dst, data[i], sizeof(float) * kInputHeight * kInputWidth * 3);
    dst += kInputHeight * kInputWidth * 3;
  }

  predictor_->run(input_tensors_, &output_tensors_);

  auto *fc8 = predictor_->ws()->GetBlob("fc8");
  CAFFE_ENFORCE(fc8, "Blob: fc8 does not exist");
  return fc8->template Get<TensorCPU>().data<float>();
}

}

