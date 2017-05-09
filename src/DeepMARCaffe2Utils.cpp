//
// Created by ken.yu on 17-5-9.
//

#include <DeepMARCaffe2.hpp>
#include <caffe2/core/common_gpu.h>

#ifndef CPU_ONLY
#include <caffe2/core/common_gpu.h>
#endif

namespace cripac {

void DeepMAR::setDevice() {
#ifndef CPU_ONLY
  SetDefaultGPUID(gpuIndex);
#endif
}

}