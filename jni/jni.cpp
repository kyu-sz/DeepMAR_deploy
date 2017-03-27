//
// Created by ken.yu on 17-3-27.
//

#include "org_cripac_isee_alg_pedestrian_attr_DeepMARCaffeNative.h"

/*
 * Class:     org_cripac_isee_alg_pedestrian_attr_DeepMARCaffeNative
 * Method:    initialize
 * Signature: (I[C[C)J
 */
JNIEXPORT jlong JNICALL Java_org_cripac_isee_alg_pedestrian_attr_DeepMARCaffeNative_initialize
    (JNIEnv *env, jobject self, jint pNet, jcharArray pbPath, jcharArray modelPath) {
}

/*
 * Class:     org_cripac_isee_alg_pedestrian_attr_DeepMARCaffeNative
 * Method:    free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_cripac_isee_alg_pedestrian_attr_DeepMARCaffeNative_free
    (JNIEnv *env, jobject self, jlong pNet) {
}

/*
 * Class:     org_cripac_isee_alg_pedestrian_attr_DeepMARCaffeNative
 * Method:    recognize
 * Signature: (J[F)[F
 */
JNIEXPORT jfloatArray JNICALL Java_org_cripac_isee_alg_pedestrian_attr_DeepMARCaffeNative_recognize
    (JNIEnv *env, jobject self, jlong pNet, jfloatArray input) {
}