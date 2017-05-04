cmake_minimum_required(VERSION 2.8.7)

project(DeepMAR_deploy CXX C)

# ---[ CMake scripts + modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
set(CMAKE_MACOSX_RPATH 1)

enable_testing()

set(CPU_ONLY $ENV{CPU_ONLY} CACHE OPTION "Whether to use CPU only")

set(CAFFE_HOME $ENV{CAFFE_HOME} CACHE PATH "Home directory of Caffe.")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
set(JAVA_HOME $ENV{JAVA_HOME} CACHE PATH "Home directory of Java Divelooment Toolkit (JDK).")

set(USE_CUDA ON BOOL "Whether to use CUDA")
set(USE_GLOG ON BOOL "Whether to use GLOG")

include(cmake/Utils.cmake)
include(cmake/Summary.cmake)

# ---[ Dependencies
include(cmake/Dependencies.cmake)

add_subdirectory(src lib)

add_subdirectory(jni lib/jni)

set(TEST_IMG_PATH CACHE FILEPATH "File path of image for test.")
add_subdirectory(test bin)

install(FILES
        ${PROJECT_SOURCE_DIR}/lib/libDeepMAR.so
        ${PROJECT_SOURCE_DIR}/lib/jni/libDeepMARCaffeNative.so
        DESTINATION
        lib)