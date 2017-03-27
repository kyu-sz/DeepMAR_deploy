cmake_minimum_required(VERSION 2.8.7)
project(DeepMAR_deploy)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set (JAVA_HOME $ENV{JAVA_HOME} CACHE PATH "Home directory of Java Divelooment Toolkit (JDK).")

link_directories(${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(jni lib/jni)

set(TEST_VIDEO_PATH CACHE FILEPATH "File path of image for test.")
add_subdirectory(test bin)

install (FILES
        ${PROJECT_SOURCE_DIR}/lib/libDeepMARCaffeNative.so
        DESTINATION
        lib)