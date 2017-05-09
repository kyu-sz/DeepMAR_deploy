add_library(DeepMARCaffe SHARED DeepMARCaffe2.cpp DeepMARCaffe2Utils.cpp)

set(CMAKE_C_CFLAGS "${CMAKE_C_CFLAGS} -fPIC")
set(CMAKE_CXX_CFLAGS "${CMAKE_CXX_CFLAGS} -fPIC")

if (${CPU_ONLY})
    message(STATUS "Compiling with CPU_ONLY!")
    add_definitions(-DCPU_ONLY)
else ()
    message(STATUS "GPU enabled!")
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
include_directories(DeepMARCaffe PUBLIC /usr/include/eigen3)

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

find_library(CAFFE2_CPU_LIB Caffe2_CPU HINTS /usr/local/lib)
target_link_libraries(DeepMARCaffe PUBLIC ${CAFFE2_CPU_LIB})

if (NOT ${CPU_ONLY})
    find_library(CAFFE2_GPU_LIB Caffe2_GPU HINTS /usr/local/lib)
    target_link_libraries(DeepMARCaffe PUBLIC ${CAFFE2_GPU_LIB})
endif ()

install(TARGETS DeepMARCaffe DESTINATION lib)