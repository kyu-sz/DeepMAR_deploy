add_library(DeepMAR_deploy SHARED attributes_recognizer.cpp)

set (CMAKE_C_CFLAGS "${CMAKE_C_CFLAGS} -fPIC")
set (CMAKE_CXX_CFLAGS "${CMAKE_CXX_CFLAGS} -fPIC")

include_directories (DeepMAR_deploy PUBLIC ${PROJECT_SOURCE_DIR}/include)
include_directories (DeepMAR_deploy PUBLIC ${JAVA_HOME}/include)

if (MSVC)
    include_directories (DeepMAR_deploy PUBLIC ${JAVA_HOME}/include/windows)
else ()
    include_directories (DeepMAR_deploy PUBLIC ${JAVA_HOME}/include/linux)
endif ()

install (TARGETS DeepMAR_deploy DESTINATION lib)