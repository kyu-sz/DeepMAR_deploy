
/** @file attributes_recognizer.cpp
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cassert>

#include <vector>
#include <string>

#include <DeepMARCaffe.hpp>
#include <caffe/caffe.hpp>

using namespace std;
using namespace caffe;

namespace cripac {

void DeepMAR::setDevice() {
  if (gpuIndex < 0) {
    Caffe::set_mode(Caffe::CPU);
  } else {
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(gpuIndex);
    Caffe::DeviceQuery();
  }
}

/**
 * Initialize the DeepMAR network with a protocol buffer file and a model file.
 * @param proto_path
 * @param model_path
 * @param gpu_index
 * @return 0 on success; negative values on failure.
 */
int DeepMAR::initialize(const char *proto_path,
                        const char *model_path,
                        int gpu_index) {
  // Check input.
  if (proto_path == nullptr) {
    fprintf(stderr, "Error: protocol buffer path is nullptr at file %s, line %d\n",
            __FILE__, __LINE__);
    fflush(stdout), fflush(stderr);
    return DEEPMAR_ILLEGAL_ARG;
  }
  // model_path input.
  if (model_path == nullptr) {
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
  fprintf(stdout, "Loading protocol from %s...\n", proto_path);
  fflush(stdout), fflush(stderr);
  net.reset(new Net<float>(proto_path, TEST));
  fprintf(stdout, "Loading caffemodel from %s...\n", model_path);
  fflush(stdout), fflush(stderr);
  net->CopyTrainedLayersFrom(model_path);

  fprintf(stdout, "Managing I/O blobs...");
  fflush(stdout), fflush(stderr);
  vector<Blob<float> *> input_blobs = net->input_blobs();
  if (input_blobs.size() == 0) {
    fflush(stdout), fflush(stderr);
    return DEEPMAR_NO_INPUT_BLOB;
  }
  input_blob = net->input_blobs()[0];
  input_blob->Reshape(1, 3, kInputHeight, kInputWidth);
  output_blob = net->blob_by_name("fc8");

  fflush(stdout), fflush(stderr);
  return DEEPMAR_OK;
}

// Fetch the data of fc8.
const float* DeepMAR::recognize(const float *input) {
  assert(input != nullptr);

  // In case this instance is used in another thread.
  setDevice();

  // Put the input into input blob.
  if (currentBatchSize != 1)
    input_blob->Reshape(currentBatchSize = 1, 3, kInputHeight, kInputWidth);
  memcpy(input_blob->mutable_cpu_data(), input, sizeof(float) * kInputHeight * kInputWidth * 3);

  net->Forward();

  return output_blob->cpu_data();
}

const float *DeepMAR::recognize(int numImages, float *data[]) {
  assert(data != nullptr);

  // In case this instance is used in another thread.
  setDevice();

  if (currentBatchSize != numImages)
    input_blob->Reshape(currentBatchSize = numImages, 3, kInputHeight, kInputWidth);

  float *dst = input_blob->mutable_cpu_data();
  for (int i = 0; i < numImages; ++i) {
    memcpy(dst, data[i], sizeof(float) * kInputHeight * kInputWidth * 3);
    dst += kInputHeight * kInputWidth * 3;
  }

  net->Forward();

  return output_blob->cpu_data();
}

}

