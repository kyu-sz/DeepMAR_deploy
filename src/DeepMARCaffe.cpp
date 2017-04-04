
/** @file attributes_recognizer.cpp
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <vector>
#include <string>

#include <DeepMARCaffe.hpp>
#include <caffe/caffe.hpp>

using namespace std;
using namespace caffe;

namespace cripac {

/**
 * Initialize the DeepMAR network with a protocol buffer file and a model file.
 * @param proto_path
 * @param model_path
 * @param gpu_index
 * @return
 */
int DeepMAR::initialize(const char *proto_path,
                        const char *model_path,
                        int gpu_index) {
  // Check input.
  if (proto_path == nullptr) {
    fprintf(stderr, "Error: protocol buffer path is nullptr at file %s, line %d\n",
            __FILE__, __LINE__);
    fflush(stdout), fflush(stderr);
    return DEEPMAR_FILE_NOT_FOUND;
  }
  // model_path input.
  if (model_path == nullptr) {
    fprintf(stderr, "Error: Caffe model path is nullptr at file %s, line %d\n",
            __FILE__, __LINE__);
    fflush(stdout), fflush(stderr);
    return DEEPMAR_FILE_NOT_FOUND;
  }

  const int kCPUOnly = -1;

  // Set mode
  if (gpu_index == kCPUOnly) {
    fprintf(stdout, "Using CPU.\n");
    fflush(stdout), fflush(stderr);
    Caffe::set_mode(Caffe::CPU);
  } else if (gpu_index >= 0) {
    fprintf(stdout, "Using GPU with device ID %d\n", gpu_index);
    fflush(stdout), fflush(stderr);
    Caffe::SetDevice(gpu_index);
    Caffe::set_mode(Caffe::GPU);
  }
  fflush(stdout), fflush(stderr);

  // Load the network.
  fprintf(stdout, "Loading protocol from %s.\n", proto_path);
  fflush(stdout), fflush(stderr);
  net.reset(new Net<float>(proto_path, TEST));
  fprintf(stdout, "Loading caffemodel from %s.\n", model_path);
  fflush(stdout), fflush(stderr);
  net->CopyTrainedLayersFrom(model_path);

  fflush(stdout), fflush(stderr);
  return DEEPMAR_OK;
}

// Fetch the data of fc8.
int DeepMAR::recognize(const float *input,
                       float *fc8) {
  // Check input.
  if (input == nullptr) {
    fprintf(stderr, "Error: input is nullptr at file %s, line %d\n", __FILE__, __LINE__);
    fflush(stdout), fflush(stderr);
    return DEEPMAR_EMPTY_INPUT;
  }

  const int kInputHeight = 227;
  const int kInputWidth = 227;

  // Put the input into input blob.
  Blob<float> *input_layer = net->input_blobs()[0];
  input_layer->Reshape(1, 3, kInputHeight, kInputWidth);
  float *input_data = input_layer->mutable_cpu_data();
  memcpy(input_data, input, sizeof(float) * input_layer->count());

  net->Forward();

  // Get input.
  boost::shared_ptr<Blob<float>> output_blob = net->blob_by_name("fc8");
  memcpy(fc8, output_blob->cpu_data(), output_blob->count() * sizeof(float));

  fflush(stdout), fflush(stderr);
  return DEEPMAR_OK;
}

}

