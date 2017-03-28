
/** @file  DeepMAR.h
 *  @brief interface for pedestrian attribute recognition using DeepMAR.
 *  @date  2017/03/27
 */

#ifndef _ATTRIBUTES_RECOGNIZER_H_
#define _ATTRIBUTES_RECOGNIZER_H_

#include <memory>

namespace caffe {
template<typename Dtype>
class Net;
}

namespace cripac {

class DeepMAR {
 private:
  std::shared_ptr<caffe::Net<float>> net;
 public:
  static const int DEEPMAR_OK = 0;
  static const int DEEPMAR_FILE_NOT_FOUND = -1;
  static const int DEEPMAR_EMPTY_INPUT = -2;

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
   *  \param[IN]  proto_filename
   *  \param[IN]  weights_filename
   *  \param[IN]  gpu_index: -1 for cpu only
   *  \param[OUT] fc8
   *  \return error code: 0 for success; <0 for fail.
   */
  int recognize(const float *data, float *fc8);

};

}

#endif  // _ATTRIBUTES_RECOGNIZER_H_
