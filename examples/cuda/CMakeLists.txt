#cmake_minimum_required(VERSION 2.8)
cmake_minimum_required(VERSION 3.1.3)
project(MatrixProduct)

if (COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
#    cmake_policy(SET CMP0057 NEW)
endif (COMMAND cmake_policy)

set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )  # not necessary, but encouraged
#set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17" )

#set( CUDA_PROPAGATE_HOST_FLAGS OFF)
#set( CUDA_SEPARABLE_COMPILATION ON)
set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11 -O3 -gencode arch=compute_20,code=sm_21 )

add_library(libvizdoom SHARED IMPORTED GLOBAL)
set_property(TARGET libvizdoom PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../build/bin/libvizdoom.so)

#find_package(OpenCV 3.1 EXACT REQUIRED PATHS /usr/local/opencv3.1)
find_package(Boost 1.74.0 EXACT REQUIRED)
find_package(CUDA QUIET REQUIRED)
find_package(Threads REQUIRED)


set( VIZDOOM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include )
set( VIZDOOM_EXAMPLES_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
set( VIZDOOM_LIBS libvizdoom ${Boost_LIBRARIES}	${CMAKE_THREAD_LIBS_INIT} )
set( VIZDOOM_LIBS ${VIZDOOM_LIBS} rt )

include_directories( ${VIZDOOM_INCLUDE_DIR} ${Boost_INCLUDE_DIR} )
# Specify include directories
#include_directories(
#    kernels
#    utility
#    )

# Specify library paths
#link_directories(
#    /opt/cuda/lib64
#    /opt/foobar/lib
#    /opt/joestuff/lib
#    )

# For compilation ...
# Specify target & source files to compile it from (matrix.h is optional)
#cuda_add_executable( main main.cpp matrix.cu )
#cuda_add_executable( main_cpp main.cpp ./agent/doomagent.cpp ./agent/actionvaluenetwork.cpp ./agent/networkelements.cpp ./agent/experiencereplaybuffer.cpp)
cuda_add_executable( main main.cpp ./agent/doomagent.cpp ./agent/experiencereplaybuffer.cpp ./agent/actionvaluenetwork.cpp ./agent/networklayer.cu ./agent/networkelements.cpp )
target_link_libraries( main ${VIZDOOM_LIBS} )
cuda_add_executable( test test.cpp ./agent/doomagent.cpp ./agent/experiencereplaybuffer.cpp ./agent/actionvaluenetwork.cpp ./agent/networklayer.cu ./agent/networkelements.cpp )
target_link_libraries( test ${VIZDOOM_LIBS} )

#set_target_properties( main
#	PROPERTIES
#	RUNTIME_OUTPUT_DIRECTORY ${VIZDOOM_EXAMPLES_OUTPUT_DIR}
#	RUNTIME_OUTPUT_DIRECTORY_DEBUG ${VIZDOOM_EXAMPLES_OUTPUT_DIR}
#	RUNTIME_OUTPUT_DIRECTORY_RELEASE ${VIZDOOM_EXAMPLES_OUTPUT_DIR}
#	RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${VIZDOOM_EXAMPLES_OUTPUT_DIR}
#	RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${VIZDOOM_EXAMPLES_OUTPUT_DIR} )

# For linking ...
# Specify target & libraries to link it with
#target_link_libraries(
#    matrix-product
#    -lcublas
#    -lfoobar
#    -ljoestuff
#    )
