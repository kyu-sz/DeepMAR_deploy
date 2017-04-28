
/** @file  DeepMAR.h
 *  @brief interface for pedestrian attribute recognition using DeepMAR.
 *  @date  2017/03/27
 */

#ifndef _ATTRIBUTES_RECOGNIZER_H_
#define _ATTRIBUTES_RECOGNIZER_H_

#include <memory>
#include <boost/shared_ptr.hpp>

namespace caffe {
template<typename Dtype>
class Net;
template<typename Dtype>
class Blob;
}

namespace cripac {

class DeepMAR {
 private:
  std::shared_ptr<caffe::Net<float>> net;
  boost::shared_ptr<caffe::Blob<float>> output_blob;
  caffe::Blob<float> *input_blob;
  const static int kInputHeight = 227;
  const static int kInputWidth = 227;

  int currentBatchSize = 1;
  // GPU index.
  int gpuIndex = -1;
 public:
  enum DeepMARStatus {
    DEEPMAR_OK = 0,
    DEEPMAR_ILLEGAL_ARG = -1,
    DEEPMAR_NO_INPUT_BLOB = -2,
  };

  const static int FC8_LEN = 1000;

  DeepMAR(void) {}
  ~DeepMAR(void) {}

  /**
   * Initialize the DeepMAR network with a protocol buffer file and a model file.
   * @param proto_filename
   * @param weights_filename
   * @param gpu_index
   * @return
   */
  int initialize(const char *proto_filename,
                 const char *weights_filename,
                 int gpu_index);

  /**
   * Recognize attributes from a single image.
   *  \param[IN]  data: 227 x 227 x 3 input
   *  \return pointer to fc8 CPU data (do not free this pointer!)
   */
  const float * recognize(const float *data);

  /**
   * Recognize attributes from an image batch.
   *  \param[IN]  numImages: number of images in the batch
   *  \param[IN]  data: numImages x 227 x 227 x 3 input
   *  \return pointer to fc8 CPU data (do not free this pointer!)
   */
  const float *recognize(int numImages, float *data[]);

};

}

#endif  // _ATTRIBUTES_RECOGNIZER_H_
