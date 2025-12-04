cmake_minimum_required(VERSION 3.10)
project(SpectralClustering)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(include)
add_executable(spectral src/main.cpp src/SpectralClustering.cpp)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(spectral_mpi src/mainMPI.cpp src/SpectralClusteringMPI.cpp)
target_link_libraries(spectral_mpi ${MPI_CXX_LIBRARIES} Eigen3::Eigen)