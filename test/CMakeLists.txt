
add_executable(DeepMAR_test test.cpp)

include_directories(DeepMAR_test PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(DeepMAR_test DeepMARCaffe opencv_core opencv_highgui opencv_imgproc)

set(TEST_IMG_PATH
        ${PROJECT_SOURCE_DIR}/test/CAM01_2014-02-15_20140215161032-20140215162620_tarid0_frame218_line1.png
        CACHE PATH "Path of the input image for test.")

set(TEST_PROTO_PATH
        ${PROJECT_SOURCE_DIR}/../../../models/DeepMARCaffe2/init_net.pb
        CACHE PATH "Path of DeepMAR protocol buffer.")

set(TEST_MODEL_PATH
        ${PROJECT_SOURCE_DIR}/../../../models/DeepMARCaffe2/predict_net.pb
        CACHE PATH "Path of DeepMAR model.")

set(TEST_GPU_ID -1 CACHE STRING "Index of GPU to use for test.")

add_test(NAME DeepMAR_test
        COMMAND DeepMAR_test
        -i ${TEST_IMG_PATH}
        -p ${TEST_PROTO_PATH}
        -m ${TEST_MODEL_PATH}
        -g ${TEST_GPU_ID})