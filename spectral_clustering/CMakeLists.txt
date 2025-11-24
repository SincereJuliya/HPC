cmake_minimum_required(VERSION 3.10)
project(SpectralClustering)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(include)
add_executable(spectral src/main.cpp src/SpectralClustering.cpp)
