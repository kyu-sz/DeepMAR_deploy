
/** @file  DeepMARCaffe2.h
 *  @brief interface for pedestrian attribute recognition using DeepMAR.
 *  @date  2017/03/27
 */

#ifndef _ATTRIBUTES_RECOGNIZER_H_
#define _ATTRIBUTES_RECOGNIZER_H_

#include <memory>
#include <vector>

namespace caffe2 {
class Workspace;
class CPUContext;
template<class T>
class Tensor;
using TensorCPU = Tensor<CPUContext>;
}

namespace cripac {

class DeepMAR {
 private:
  std::unique_ptr<caffe2::Workspace> workspace_;
  const static int kInputHeight = 227;
  const static int kInputWidth = 227;

  std::string network_name_;

  caffe2::TensorCPU *input_buf_;
  caffe2::TensorCPU *output_buf_;

  int current_batch_size_ = 0;
 public:
  enum DeepMARStatus {
    DEEPMAR_OK = 0,
    DEEPMAR_ILLEGAL_ARG = -1,
    DEEPMAR_NO_INPUT_BLOB = -2,
  };

  const static int FC8_LEN = 1000;

  DeepMAR();
  ~DeepMAR();

  /**
   * Initialize the DeepMAR network with a protocol buffer file and a model file.
   * @param init_net_path
   * @param weights_filename
   * @param gpu_index
   * @return
   */
  int initialize(const char *init_net_path,
                 const char *predict_net_path,
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
  const float *recognize(int numImages, const float *data[]);

};

}

#endif  // _ATTRIBUTES_RECOGNIZER_H_
