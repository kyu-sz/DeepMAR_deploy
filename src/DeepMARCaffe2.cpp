
/** @file attributes_recognizer.cpp
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cassert>

#include <vector>
#include <string>

#include <DeepMARCaffe2.hpp>
#include <caffe2/core/init.h>
#include <caffe2/core/predictor.h>
#include <caffe2/utils/proto_utils.h>

#ifndef CPU_ONLY
#include <caffe2/core/context_gpu.h>
#endif

using namespace std;
using namespace caffe2;

namespace cripac {

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

  // Load the network.
  fprintf(stdout, "Loading protocol from %s...\n", init_net_path);
  fflush(stdout), fflush(stderr);
  NetDef init_net, predict_net;
  CAFFE_ENFORCE(ReadProtoFromFile(init_net_path, &init_net));
  CAFFE_ENFORCE(ReadProtoFromFile(predict_net_path, &predict_net));
  VLOG(1) << "Init net: " << ProtoDebugString(init_net);
  LOG(INFO) << "Predict net: " << ProtoDebugString(predict_net);

  // Set device.
  DeviceOption dev_opt;
  if (gpu_index < 0) {
    fprintf(stdout, "Using CPU.\n");
    fflush(stdout);
    dev_opt.set_device_type(caffe2::CPU);
  } else {
    fprintf(stdout, "Using device %d.\n", gpu_index);
    fflush(stdout);
    dev_opt.set_device_type(caffe2::CUDA);
    dev_opt.set_cuda_gpu_id(gpu_index);
  }
  init_net.mutable_device_option()->CopyFrom(dev_opt);
  predict_net.mutable_device_option()->CopyFrom(dev_opt);
  network_name_ = predict_net.name();

  // Create predictor.
  CAFFE_ENFORCE(workspace_->RunNetOnce(init_net));
  CAFFE_ENFORCE(workspace_->CreateNet(predict_net));

  fflush(stdout), fflush(stderr);
  return DEEPMAR_OK;
}

DeepMAR::DeepMAR() {
  workspace_ = make_unique<Workspace>();

  input_buf_ = new TensorCPU;
#ifndef CPU_ONLY
  output_buf_ = new TensorCPU;
#endif

  input_buf_->Resize(current_batch_size_ = 1, 3, kInputHeight, kInputWidth);
}

DeepMAR::~DeepMAR() {
  delete (input_buf_);
#ifndef CPU_ONLY
  delete (output_buf_);
#endif
}

const float *DeepMAR::recognize(const float *input) {
  assert(input != nullptr);
  const float *inputs[] = {input};
  return recognize(sizeof(inputs) / sizeof(const float *), inputs);
}

const float *DeepMAR::recognize(int numImages, const float *inputs[]) {
  assert(inputs != nullptr);

  // Fill the data into the input tensor buffer.
  if (current_batch_size_ != numImages)
    input_buf_->Resize(current_batch_size_ = numImages, 3, kInputHeight, kInputWidth);
  float *ptr = input_buf_->mutable_data<float>();
  for (int i = 0; i < numImages; ++i, ptr += kInputHeight * kInputWidth * 3) {
    assert(inputs[i] != nullptr);
    memcpy(ptr, inputs[i], sizeof(float) * kInputHeight * kInputWidth * 3);
  }

  // Copy data from the buffer to the blob in the network.
  auto *blob = workspace_->GetBlob("data");
  CAFFE_ENFORCE(blob, "Blob: ", "data", " does not exist");
  if (blob->IsType<TensorCPU>())
    blob->template GetMutable<TensorCPU>()->CopyFrom(*input_buf_);
  else
#ifdef CPU_ONLY
    throw new std::logic_error("DeepMARCaffe2 is compiled with no GPU support, but the network is built on GPU!");
#else
    blob->template GetMutable<TensorCUDA>()->CopyFrom(*input_buf_);
#endif

  // Forward.
  workspace_->RunNet(network_name_);

  // Get the results.
  auto *fc8 = workspace_->GetBlob("fc8");
  CAFFE_ENFORCE(fc8, "Blob: fc8 does not exist");
  if (fc8->IsType<TensorCPU>())
    return fc8->Get<TensorCPU>().data<float>();
  else {
#ifdef CPU_ONLY
    throw new std::logic_error("DeepMARCaffe2 is compiled with no GPU support, but the network is built on GPU!");
#else
    // Need a transfer from GPU to CPU.
    output_buf_->CopyFrom(fc8->Get<TensorCUDA>());
    return output_buf_->data<float>();
#endif
  }
}
}

