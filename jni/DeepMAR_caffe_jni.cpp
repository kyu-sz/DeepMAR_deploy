//
// Created by ken.yu on 17-3-27.
//

#include <DeepMAR.h>
#include <DeepMAR_caffe_jni.h>

using namespace cripac;

/*
 * Class:     org_cripac_isee_alg_pedestrian_attr_DeepMARCaffeNative
 * Method:    initialize
 * Signature: (I[B[B)J
 */
JNIEXPORT jlong JNICALL Java_org_cripac_isee_alg_pedestrian_attr_DeepMARCaffeNative_initialize
    (JNIEnv *env, jobject self, jint gpu_id, jbyteArray pb_path, jbyteArray model_path) {
  DeepMAR *deepMAR = new DeepMAR();
  deepMAR->initialize((const char *) env->GetByteArrayElements(pb_path, nullptr),
                      (const char *) env->GetByteArrayElements(model_path, nullptr),
                      gpu_id);
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
  float output_c[1024];
  deepMAR->recognize(env->GetFloatArrayElements(input, nullptr), output_c);
  env->SetFloatArrayRegion(output, 0, env->GetArrayLength(output), output_c);
}