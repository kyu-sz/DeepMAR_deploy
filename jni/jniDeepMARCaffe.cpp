//
// Created by ken.yu on 17-3-27.
//

#include <DeepMARCaffe.hpp>
#include <jniDeepMARCaffe.h>

using namespace cripac;

/*
 * Class:     org_cripac_isee_alg_pedestrian_attr_DeepMARCaffeNative
 * Method:    initialize
 * Signature: (I[B[B)J
 */
JNIEXPORT jlong JNICALL Java_org_cripac_isee_alg_pedestrian_attr_DeepMARCaffeNative_initialize
    (JNIEnv *env, jobject self, jint gpu_id, jstring j_pb_path, jstring j_model_path) {
  DeepMAR *deepMAR = new DeepMAR();

  const int pb_len = env->GetStringUTFLength(j_pb_path);
  const int model_len = env->GetStringUTFLength(j_model_path);
  char *c_pb_path = new char[pb_len + 1];
  char *c_model_path = new char[model_len + 1];
  env->GetStringUTFRegion(j_pb_path, 0, pb_len, c_pb_path);
  env->GetStringUTFRegion(j_model_path, 0, model_len, c_model_path);
  c_pb_path[pb_len] = '\0';
  c_model_path[model_len] = '\0';

  deepMAR->initialize(c_pb_path, c_model_path, gpu_id);

  delete[](c_model_path);
  delete[](c_pb_path);

  return (jlong) deepMAR;
}

/*
 * Class:     org_cripac_isee_alg_pedestrian_attr_DeepMARCaffeNative
 * Method:    free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_cripac_isee_alg_pedestrian_attr_DeepMARCaffeNative_free
    (JNIEnv *env, jobject self, jlong net) {
  free((DeepMAR *) net);
}

/*
 * Class:     org_cripac_isee_alg_pedestrian_attr_DeepMARCaffeNative
 * Method:    recognize
 * Signature: (J[F[F)V
 */
JNIEXPORT void JNICALL Java_org_cripac_isee_alg_pedestrian_attr_DeepMARCaffeNative_recognize
    (JNIEnv *env, jobject self, jlong net, jfloatArray j_input, jfloatArray output) {
  DeepMAR* deepMAR = (DeepMAR *) net;
  float* c_input = env->GetFloatArrayElements(j_input, nullptr);
  env->SetFloatArrayRegion(output, 0, env->GetArrayLength(output), deepMAR->recognize(c_input));
  env->ReleaseFloatArrayElements(j_input, c_input, 0);
}