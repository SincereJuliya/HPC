# ==========================================
# Makefile for HPC Spectral Clustering
# ==========================================

# Compilers
# UPDATED: Use mpicxx for everything to ensure C++17 support on cluster
CXX      = mpicxx
MPICXX   = mpicxx

# Flags
# -O3: Maximum optimization
# -std=c++17: Required language standard
# -fopenmp: Required for Hybrid Parallelism (OpenMP)
# -I./eigen_local: Looks for Eigen in the project folder
# -I/usr/include/eigen3: System backup
CXXFLAGS = -O3 -std=c++17 -fopenmp -I./eigen_local -I/usr/include/eigen3

# Output Binaries
TARGET_SEQ = spectral_seq
TARGET_MPI = spectral_mpi

# Default Target
all: $(TARGET_SEQ) $(TARGET_MPI)

# ------------------------------------------
# Sequential Build
# ------------------------------------------
$(TARGET_SEQ): src/main.cpp src/SpectralClustering.cpp src/SpectralClustering.hpp
	@echo "Compiling Sequential Version..."
	$(CXX) $(CXXFLAGS) -o $(TARGET_SEQ) src/main.cpp src/SpectralClustering.cpp

# ------------------------------------------
# Parallel Build (MPI + OpenMP)
# ------------------------------------------
$(TARGET_MPI): src/mainMPI.cpp src/SpectralClusteringMPI.cpp src/SpectralClusteringMPI.hpp
	@echo "Compiling MPI Version..."
	$(MPICXX) $(CXXFLAGS) -o $(TARGET_MPI) src/mainMPI.cpp src/SpectralClusteringMPI.cpp

# ------------------------------------------
# Clean Up
# ------------------------------------------
clean:
	@echo "Cleaning up..."
	rm -f $(TARGET_SEQ) $(TARGET_MPI) *.o

.PHONY: all clean