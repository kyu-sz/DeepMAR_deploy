
/** @file attributes_recognizer.cpp
 */

#include "attributes_recognizer.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <vector>
#include <string>
using namespace std;

#include <caffe/caffe.hpp>
using namespace caffe;

// Fetch the data of fc8.
int AttributeRecognizer::recognize(const float* data, const char* proto_filename, 
    const char* weights_filename, int gpu_index, float* fc8) {

    // Input Checking.
    if (data == NULL) {
        fprintf(stderr, "Error: input data is NULL in file %s, line %d\n", 
                __FILE__, __LINE__);
        return -1;
    }
    if (proto_filename == NULL) {
        fprintf(stderr, "Error: filename of proto file is NULL in file %s, line %d\n",
                __FILE__, __LINE__);
        return -2;
    }
    if (weights_filename == NULL) {
        fprintf(stderr, "Error: filename of caffmodel is NULL in file %s, line %d\n",
                __FILE__, __LINE__);
        return -2;
    }

    const int kCpuOnly = -1;
    const int kInputHeight = 227;
    const int kInputWidth = 227;
    // Set mode
    if (gpu_index == kCpuOnly) {
        fprintf(stdout, "Use CPU.\n");
        Caffe::set_mode(Caffe::CPU);
    } else if (gpu_index >= 0) {
        fprintf(stdout, "Use GPU with device ID %d\n", gpu_index);
        Caffe::SetDevice(gpu_index);
        Caffe::set_mode(Caffe::GPU);
    }

    // Load the network.
    string proto_file = proto_filename;
    string caffemodel_file = weights_filename;

    boost::shared_ptr<Net<float> > net;
    net.reset(new Net<float>(proto_file, TEST));
    net->CopyTrainedLayersFrom(caffemodel_file);

    // Put the data into data blob.
    Blob<float>* input_layer = net->input_blobs()[0];
    input_layer->Reshape(1, 3, kInputHeight, kInputWidth);
    float* input_data = input_layer->mutable_cpu_data();
    memcpy(input_data, data, sizeof(float)*input_layer->count());

    net->Forward();

    // Get data.
    boost::shared_ptr<Blob<float> > output_blob = net->blob_by_name("fc8");
    fc8 = new float[output_blob->count()];
    memcpy(fc8, output_blob->cpu_data(), output_blob->count());
    return 0;
}

