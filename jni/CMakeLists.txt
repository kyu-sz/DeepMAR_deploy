add_library(jniDeepMARCaffe2 SHARED jniDeepMARCaffe2.cpp)

set (CMAKE_C_CFLAGS "${CMAKE_C_CFLAGS} -fPIC")
set (CMAKE_CXX_CFLAGS "${CMAKE_CXX_CFLAGS} -fPIC")

include_directories(jniDeepMARCaffe2 PUBLIC ${PROJECT_SOURCE_DIR}/include)
include_directories(jniDeepMARCaffe2 PUBLIC ${JAVA_HOME}/include)

target_link_libraries(jniDeepMARCaffe2 DeepMARCaffe2 opencv_core opencv_highgui opencv_imgproc)

if (MSVC)
    target_include_directories(jniDeepMARCaffe2 PUBLIC ${JAVA_HOME}/include/windows)
else ()
    target_include_directories(jniDeepMARCaffe2 PUBLIC ${JAVA_HOME}/include/linux)
endif ()

install(TARGETS jniDeepMARCaffe2 DESTINATION lib)