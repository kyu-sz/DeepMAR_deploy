//
// Created by ken.yu on 17-3-27.
//

#include <DeepMARCaffe2.hpp>
#include <jniDeepMARCaffe2.h>

using namespace cripac;

/*
 * Class:     org_cripac_isee_alg_pedestrian_attr_DeepMARCaffe2Native
 * Method:    initialize
 * Signature: (I[B[B)J
 */
JNIEXPORT jlong JNICALL Java_org_cripac_isee_alg_pedestrian_attr_DeepMARCaffe2Native_initialize
    (JNIEnv *env, jobject self, jint gpu_id, jstring j_init_net_path, jstring j_predict_path) {
  DeepMAR *deepMAR = new DeepMAR();

  const int init_net_path_len = env->GetStringUTFLength(j_init_net_path);
  const int predict_net_path_len = env->GetStringUTFLength(j_predict_path);
  char *c_init_net_path = new char[init_net_path_len + 1];
  char *c_predict_net_path = new char[predict_net_path_len + 1];
  env->GetStringUTFRegion(j_init_net_path, 0, init_net_path_len, c_init_net_path);
  env->GetStringUTFRegion(j_predict_path, 0, predict_net_path_len, c_predict_net_path);
  c_init_net_path[init_net_path_len] = '\0';
  c_predict_net_path[predict_net_path_len] = '\0';

  assert(c_init_net_path != nullptr);
  assert(c_predict_net_path != nullptr);
  assert(init_net_path_len > 0);
  assert(predict_net_path_len > 0);
  int ret = deepMAR->initialize(c_init_net_path, c_predict_net_path, gpu_id);
  if (ret != DeepMAR::DEEPMAR_OK) {
    static const char *className = "java/lang/RuntimeException";
    jclass exClass = env->FindClass(className);
    assert(exClass != nullptr);
    switch (ret) {
      case DeepMAR::DEEPMAR_NO_INPUT_BLOB:return env->ThrowNew(exClass, "No input blob found!");
      default:break;
    }
  }

  delete[](c_predict_net_path);
  delete[](c_init_net_path);

  return (jlong) deepMAR;
}

/*
 * Class:     org_cripac_isee_alg_pedestrian_attr_DeepMARCaffe2Native
 * Method:    free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_cripac_isee_alg_pedestrian_attr_DeepMARCaffe2Native_free
    (JNIEnv *env, jobject self, jlong net) {
  free((DeepMAR *) net);
}

/*
 * Class:     org_cripac_isee_alg_pedestrian_attr_DeepMARCaffe2Native
 * Method:    recognize
 * Signature: (J[F[F)V
 */
JNIEXPORT void JNICALL Java_org_cripac_isee_alg_pedestrian_attr_DeepMARCaffe2Native_recognize__J_3F_3F
    (JNIEnv *env, jobject self, jlong net, jfloatArray j_input, jfloatArray output) {
  DeepMAR *deepMAR = (DeepMAR *) net;
  float *c_input = env->GetFloatArrayElements(j_input, nullptr);
  env->SetFloatArrayRegion(output, 0, env->GetArrayLength(output), deepMAR->recognize(c_input));
  env->ReleaseFloatArrayElements(j_input, c_input, 0);
}

JNIEXPORT void JNICALL Java_org_cripac_isee_alg_pedestrian_attr_DeepMARCaffe2Native_recognize__J_3_3F_3_3F
    (JNIEnv *env, jobject self, jlong net, jobjectArray j_input, jobjectArray output) {
  DeepMAR *deepMAR = (DeepMAR *) net;

  int num_images = env->GetArrayLength(j_input);
  float **inputs = new float *[num_images];
  for (int i = 0; i < num_images; ++i)
    inputs[i] = env->GetFloatArrayElements((jfloatArray) env->GetObjectArrayElement(j_input, i), nullptr);

  const float *fc8 = deepMAR->recognize(num_images, inputs);
  for (int i = 0; i < num_images; ++i) {
    jfloatArray slice = (jfloatArray) env->GetObjectArrayElement(output, i);
    env->SetFloatArrayRegion(slice, 0, env->GetArrayLength(slice), fc8);
    fc8 += DeepMAR::FC8_LEN;
  }

  for (int i = 0; i < num_images; ++i)
    env->ReleaseFloatArrayElements((jfloatArray) env->GetObjectArrayElement(j_input, i), inputs[i], 0);
  delete[] inputs;
}