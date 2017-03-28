add_library(DeepMAR_caffe_jni SHARED DeepMAR_caffe_jni.cpp)

set (CMAKE_C_CFLAGS "${CMAKE_C_CFLAGS} -fPIC")
set (CMAKE_CXX_CFLAGS "${CMAKE_CXX_CFLAGS} -fPIC")

include_directories (DeepMAR_caffe_jni PUBLIC ${PROJECT_SOURCE_DIR}/include)
include_directories (DeepMAR_caffe_jni PUBLIC ${JAVA_HOME}/include)

target_link_libraries(DeepMAR_caffe_jni DeepMAR opencv_core opencv_highgui opencv_imgproc)

if (MSVC)
    include_directories (DeepMAR_caffe_jni PUBLIC ${JAVA_HOME}/include/windows)
else ()
    include_directories (DeepMAR_caffe_jni PUBLIC ${JAVA_HOME}/include/linux)
endif ()

install (TARGETS DeepMAR_caffe_jni DESTINATION lib)