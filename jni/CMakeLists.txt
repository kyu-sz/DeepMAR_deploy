add_library(DeepMARNativeCaffe SHARED jni.cpp)

set (CMAKE_C_CFLAGS "${CMAKE_C_CFLAGS} -fPIC")
set (CMAKE_CXX_CFLAGS "${CMAKE_CXX_CFLAGS} -fPIC")

include_directories (DeepMARNativeCaffe PUBLIC ${PROJECT_SOURCE_DIR}/include)
include_directories (DeepMARNativeCaffe PUBLIC ${JAVA_HOME}/include)

if (MSVC)
    include_directories (DeepMARNativeCaffe PUBLIC ${JAVA_HOME}/include/windows)
else ()
    include_directories (DeepMARNativeCaffe PUBLIC ${JAVA_HOME}/include/linux)
endif ()

install (TARGETS DeepMARNativeCaffe DESTINATION lib)