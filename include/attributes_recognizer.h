
/** @file  attributes_recognizer.h
 *  @brief interface for attributes recognition using DeepMar.
 *  @date  2017/03/27
 */

#ifndef _ATTRIBUTES_RECOGNIZER_H_
#define _ATTRIBUTES_RECOGNIZER_H_

class AttributeRecognizer {
public:
    AttributeRecognizer(void) {}
    ~AttributeRecognizer(void) {}

    /**
     *  \param[IN]  data
     *  \param[IN]  proto_filename
     *  \param[IN]  weights_filename
     *  \param[IN]  gpu_index: -1 for cpu only
     *  \param[OUT] fc8
     *  \return error code: 0 for success; <0 for fail.
     */
    int recognize(const float* data, const char* proto_filename,
        const char* weights_filename, int gpu_index, float* fc8);

};

#endif  // _ATTRIBUTES_RECOGNIZER_H_
