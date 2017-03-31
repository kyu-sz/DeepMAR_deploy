add_library(jniDeepMARCaffe SHARED jniDeepMARCaffe.cpp)

set (CMAKE_C_CFLAGS "${CMAKE_C_CFLAGS} -fPIC")
set (CMAKE_CXX_CFLAGS "${CMAKE_CXX_CFLAGS} -fPIC")

include_directories (jniDeepMARCaffe PUBLIC ${PROJECT_SOURCE_DIR}/include)
include_directories (jniDeepMARCaffe PUBLIC ${JAVA_HOME}/include)

target_link_libraries(jniDeepMARCaffe DeepMARCaffe opencv_core opencv_highgui opencv_imgproc)

if (MSVC)
    include_directories (jniDeepMARCaffe PUBLIC ${JAVA_HOME}/include/windows)
else ()
    include_directories (jniDeepMARCaffe PUBLIC ${JAVA_HOME}/include/linux)
endif ()

install (TARGETS jniDeepMARCaffe DESTINATION lib)