add_library(DeepMARCaffe SHARED DeepMARCaffe.cpp)

set(CMAKE_C_CFLAGS "${CMAKE_C_CFLAGS} -fPIC")
set(CMAKE_CXX_CFLAGS "${CMAKE_CXX_CFLAGS} -fPIC")

if (${CPU_ONLY})
    message(STATUS "Compiling with CPU_ONLY!")
    add_definitions(-DCPU_ONLY)
endif ()

execute_process(
        WORKING_DIRECTORY ${CAFFE_HOME}
        COMMAND protoc src/caffe/proto/caffe.proto --cpp_out=.
)
execute_process(
        COMMAND mkdir -p ${CAFFE_HOME}/include/caffe/proto
)
execute_process(
        WORKING_DIRECTORY ${CAFFE_HOME}
        COMMAND cp src/caffe/proto/caffe.pb.h include/caffe/proto
)

include_directories(DeepMARCaffe PUBLIC ${PROJECT_SOURCE_DIR}/include)
include_directories(DeepMARCaffe PUBLIC ${JAVA_HOME}/include)
include_directories(DeepMARCaffe PUBLIC ${CAFFE_HOME}/include)

if (MSVC)
    include_directories(DeepMARCaffe PUBLIC ${JAVA_HOME}/include/windows)
else ()
    include_directories(DeepMARCaffe PUBLIC ${JAVA_HOME}/include/linux)
endif ()

set(CUDA_HOME $ENV{CUDA_HOME})
if (CUDA_HOME)
    message(STATUS "Using CUDA at ${CUDA_HOME}")
    include_directories(DeepMARCaffe PUBLIC ${CUDA_HOME}/include)
    target_link_libraries(DeepMARCaffe PUBLIC ${CUDA_HOME}/lib64)
endif ()

find_library(CAFFE_LIB caffe HINTS ${CAFFE_HOME}/.build_release/lib)
target_link_libraries(DeepMARCaffe PUBLIC ${CAFFE_LIB})

install(TARGETS DeepMARCaffe DESTINATION lib)