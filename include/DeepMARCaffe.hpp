
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
  float *input_data;
  const int kInputHeight = 227;
  const int kInputWidth = 227;
 public:
  static const int DEEPMAR_OK = 0;
  static const int DEEPMAR_FILE_NOT_FOUND = -1;

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
   *  \param[IN]  data: 227x227x3 input
   *  \return pointer to fc8 CPU data (do not free this pointer!)
   */
  const float * recognize(const float *data);

};

}

#endif  // _ATTRIBUTES_RECOGNIZER_H_
