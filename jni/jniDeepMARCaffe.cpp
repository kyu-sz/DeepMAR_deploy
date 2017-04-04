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
    (JNIEnv *env, jobject self, jint gpu_id, jstring pb_path, jstring model_path) {
  DeepMAR *deepMAR = new DeepMAR();
  const char* c_pb_path = env->GetStringUTFChars(pb_path, nullptr);
  const char* c_model_path = env->GetStringUTFChars(model_path, nullptr);
  deepMAR->initialize(c_pb_path, c_model_path, gpu_id);
  env->ReleaseStringUTFChars(pb_path, c_pb_path);
  env->ReleaseStringUTFChars(model_path, c_model_path);
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
    (JNIEnv *env, jobject self, jlong net, jfloatArray input, jfloatArray output) {
  DeepMAR* deepMAR = (DeepMAR *) net;
  env->SetFloatArrayRegion(output, 0, env->GetArrayLength(output),
                           deepMAR->recognize(env->GetFloatArrayElements(input, nullptr)));
}