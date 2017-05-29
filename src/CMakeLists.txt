add_library(DeepMARCaffe2 SHARED DeepMARCaffe2.cpp)

set(CMAKE_C_CFLAGS "${CMAKE_C_CFLAGS} -fPIC")
set(CMAKE_CXX_CFLAGS "${CMAKE_CXX_CFLAGS} -fPIC")

if (${CPU_ONLY})
    message(STATUS "Compiling with CPU_ONLY!")
    add_definitions(-DCPU_ONLY)
else ()
    message(STATUS "GPU enabled!")
endif ()

target_include_directories(DeepMARCaffe2 PUBLIC ${CAFFE2_INSTALL_HOME}/include)
target_include_directories(DeepMARCaffe2 PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(DeepMARCaffe2 PUBLIC ${JAVA_HOME}/include)
target_include_directories(DeepMARCaffe2 PUBLIC /usr/include/eigen3)

if (MSVC)
    target_include_directories(DeepMARCaffe2 PUBLIC ${JAVA_HOME}/include/windows)
else ()
    target_include_directories(DeepMARCaffe2 PUBLIC ${JAVA_HOME}/include/linux)
endif ()

set(CUDA_HOME $ENV{CUDA_HOME})
if (CUDA_HOME)
    message(STATUS "Using CUDA at ${CUDA_HOME}")
    target_include_directories(DeepMARCaffe2 PUBLIC ${CUDA_HOME}/include)
    link_directories(DeepMARCaffe2 PUBLIC ${CUDA_HOME}/lib64)
endif ()

find_library(CAFFE2_CPU_LIB Caffe2_CPU HINTS ${CAFFE2_INSTALL_HOME}/lib)
target_link_libraries(DeepMARCaffe2 PUBLIC ${CAFFE2_CPU_LIB})

if (NOT ${CPU_ONLY})
    find_library(CAFFE2_GPU_LIB Caffe2_GPU HINTS ${CAFFE2_INSTALL_HOME}/lib)
    find_library(NCCL_LIB nccl HINTS ${CAFFE2_INSTALL_HOME}/lib)
    target_link_libraries(DeepMARCaffe2 PUBLIC ${CAFFE2_GPU_LIB} ${NCCL_LIB})
endif ()

install(TARGETS DeepMARCaffe2 DESTINATION lib)
