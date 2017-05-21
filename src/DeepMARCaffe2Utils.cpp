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
  printf("DeepMAR is compiled with CPU only.\n");
#ifndef CPU_ONLY
  printf("Using GPU %d!\n", gpu_index_);
  caffe2::SetDefaultGPUID(gpu_index_);
#endif
}

}