#pragma once
#include <Eigen/Dense>
#include <vector>
#include <mpi.h>

/**
 * @brief Type definition for Row-Major Matrices.
 * * @details MPI requires data to be contiguous in memory for efficient 
 * row-based scattering/gathering. Default Eigen matrices are Column-Major, 
 * which breaks this requirement. We explicitly enforce RowMajor layout here.
 */
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixRowMajor;

/**
 * @class SpectralClusteringMPI
 * @brief High-Performance Distributed Spectral Clustering (MPI + OpenMP).
 *
 * @details This class implements a scalable Spectral Clustering algorithm designed 
 * for HPC clusters. It overcomes the O(N^2) memory bottleneck of standard 
 * implementations by using a distributed Nyström Approximation.
 *
 * Key HPC Features:
 * - **Hybrid Parallelism:** Uses MPI for inter-node communication and OpenMP for intra-node threading.
 * - **Memory Efficiency:** Implements a Row-Block decomposition where each rank holds only a slice of data.
 * - **Nyström Method:** Approximates the similarity matrix using global landmarks to scale to N=2,000,000+.
 * - **Distributed K-Means:** Custom implementation with MPI_Allreduce for centroid synchronization.
 */
class SpectralClusteringMPI {
public:
    /**
     * @brief Constructor for the Distributed Spectral Clustering model.
     * * @param k Number of clusters to find.
     * @param knn Number of nearest neighbors (used for local scaling sigma adaptation).
     * @param sigma Global Gaussian kernel width (if fixed). Set to -1.0 to enable Self-Tuning.
     * @param kmeans_runs Number of K-Means restarts (default 5). 
     * Note: In Nyström mode, a single optimized run is often sufficient due to high separability.
     */
    SpectralClusteringMPI(int k=3, int knn=10, double sigma=-1.0, int kmeans_runs=5);

    /**
     * @brief Executes the distributed clustering pipeline.
     * * @details The pipeline consists of:
     * 1. Global Standardization (MPI_Allreduce).
     * 2. Distributed Nyström Approximation (Landmark selection & broadcasting).
     * 3. Local Projection into Spectral Space (Zero-network traffic).
     * 4. Distributed K-Means Clustering.
     * * @param localData Reference to the local data slice (Row-Major). 
     * This avoids data copying and ensures MPI compatibility.
     * @param N_global Total number of points across all MPI ranks (used for normalization).
     */
    void fit(Eigen::Ref<MatrixRowMajor> localData, int N_global);

    /**
     * @brief Returns the cluster labels assigned to the local data points.
     * @return const std::vector<int>& Vector of labels.
     */
    const std::vector<int>& get_labels() const { return labels_; }

private:
    int k_;             ///< Number of clusters
    int knn_;           ///< Neighbors for local scaling
    double sigma_;      ///< Kernel width
    int kmeans_runs_;   ///< Max K-Means iterations/runs
    std::vector<int> labels_; ///< Final cluster assignments

    /**
     * @brief Standardizes data globally without gathering it on a single node.
     * * @details Computes global Mean and StdDev using MPI_Allreduce to aggregate 
     * partial sums and squared sums from all ranks.
     * * @param localData Local data block to standardize in-place.
     * @param N_global Total dataset size.
     */
    void standardize_distributed(Eigen::Ref<MatrixRowMajor> localData, int N_global);

    /**
     * @brief Performs Distributed K-Means Clustering.
     * * @details 
     * - **Local Step:** Each rank assigns its points to the nearest centroid (OpenMP).
     * - **Global Step:** Centroids are updated via MPI_Allreduce (summing coordinates).
     * * @param localX The data projected into the spectral embedding space (local slice).
     * @param N_global Total points (for weighting).
     * @param rows_count Helper vector containing the number of rows per rank (for Gatherv).
     */
    void kmeans_hpc(const Eigen::MatrixXd& localX, int N_global, const std::vector<int>& rows_count);
};
