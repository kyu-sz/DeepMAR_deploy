enable_testing()

add_executable(DeepMAR_deploy_test EXCLUDE_FROM_ALL test.cpp)
include_directories(DeepMAR_deploy_test PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(DeepMAR_deploy_test basic_DeepMAR_deploy opencv_core opencv_highgui opencv_imgproc)
add_test(NAME DeepMAR_deploy_test COMMAND DeepMAR_deploy_test ${TEST_VIDEO_PATH})
