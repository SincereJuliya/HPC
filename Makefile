# Compilers
# UPDATED: Use mpicxx for everything to ensure C++17 support on cluster
CXX      = mpicxx
MPICXX   = mpicxx

# Flags
# -O3: Maximum optimization
# -std=c++17: Required language standard
# -fopenmp: Required for Hybrid Parallelism (OpenMP) and to resolve OpenMP symbols
# -I./eigen_local: Looks for Eigen in the project folder
# -I/usr/include/eigen3: System backup
BASE_FLAGS = -O3 -std=c++17 -I./eigen_local -I/usr/include/eigen3
OMP_FLAGS  = -fopenmp

# Output Binaries
TARGET_SEQ    = spectral_seq
TARGET_MPI    = spectral_mpi
TARGET_HYBRID = spectral_hybrid

# Default Target
all: $(TARGET_SEQ) $(TARGET_MPI) $(TARGET_HYBRID)

# ------------------------------------------
# Sequential Build
# ------------------------------------------
$(TARGET_SEQ): src/main.cpp src/SpectralClustering.cpp
	@echo "Compiling Sequential Version..."
	$(CXX) $(BASE_FLAGS) -o $(TARGET_SEQ) src/main.cpp src/SpectralClustering.cpp

# ------------------------------------------
# Parallel Build (Pure MPI)
# Note: We include OMP_FLAGS here to avoid "undefined reference" 
# to OpenMP functions used in the source code.
# ------------------------------------------
$(TARGET_MPI): src/mainMPI.cpp src/SpectralClusteringMPI.cpp
	@echo "Compiling MPI Version..."
	$(MPICXX) $(BASE_FLAGS) $(OMP_FLAGS) -o $(TARGET_MPI) src/mainMPI.cpp src/SpectralClusteringMPI.cpp

# ------------------------------------------
# Parallel Build (Hybrid: MPI + OpenMP)
# ------------------------------------------
$(TARGET_HYBRID): src/mainMPI.cpp src/SpectralClusteringMPI.cpp
	@echo "Compiling Hybrid Version..."
	$(MPICXX) $(BASE_FLAGS) $(OMP_FLAGS) -o $(TARGET_HYBRID) src/mainMPI.cpp src/SpectralClusteringMPI.cpp

# ------------------------------------------
# Clean Up
# ------------------------------------------
clean:
	@echo "Cleaning up..."
	rm -f $(TARGET_SEQ) $(TARGET_MPI) $(TARGET_HYBRID) *.o

.PHONY: all clean