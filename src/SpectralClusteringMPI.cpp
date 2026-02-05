#include "SpectralClusteringMPI.hpp"
#include <Eigen/Eigenvalues>
#include <omp.h>
#include <iostream>
#include <random>
#include <cmath>

using namespace Eigen;

SpectralClusteringMPI::SpectralClusteringMPI(int k, int knn, double sigma, int kmeans_runs)
    : k_(k), knn_(knn), sigma_(sigma), kmeans_runs_(kmeans_runs) { }

void SpectralClusteringMPI::standardize_distributed(Eigen::Ref<MatrixRowMajor> localData, int N_global) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int dim = (int)localData.cols();

    // Compute global mean
    VectorXd local_sum = localData.colwise().sum();
    VectorXd global_sum(dim);
    MPI_Allreduce(local_sum.data(), global_sum.data(), dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    VectorXd mean = global_sum / (double)N_global;

    // Compute global stddev
    VectorXd local_sq_diff = (localData.rowwise() - mean.transpose()).array().square().colwise().sum();
    VectorXd global_sq_diff(dim);
    MPI_Allreduce(local_sq_diff.data(), global_sq_diff.data(), dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    VectorXd stddev = (global_sq_diff / (double)(N_global - 1)).array().sqrt();
    for(int i=0; i<dim; ++i) if(stddev(i) < 1e-8) stddev(i) = 1.0;

    // Apply scaling
    #pragma omp parallel for schedule(static)
    for(int i=0; i<localData.rows(); ++i)
        localData.row(i) = (localData.row(i) - mean.transpose()).array() / stddev.transpose().array();
}

void SpectralClusteringMPI::fit(Eigen::Ref<MatrixRowMajor> localData, int N_global) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Enable Eigen internal parallelism for the Rank 0 decomposition
    initParallel();
    setNbThreads(omp_get_max_threads());

    int local_n = (int)localData.rows();
    int dim = (int)localData.cols();

    // 1. Global Standardize
    standardize_distributed(localData, N_global);

    // 2. Nyström Landmarks: Collect s points globally
    int s_total = 2048; 
    int s_per_rank = s_total / size;
    MatrixRowMajor local_landmarks(s_per_rank, dim);
    std::mt19937 gen(42 + rank);
    for(int i=0; i<s_per_rank; ++i)
        local_landmarks.row(i) = localData.row(gen() % local_n);

    MatrixRowMajor landmarks(s_total, dim);
    MPI_Allgather(local_landmarks.data(), s_per_rank * dim, MPI_DOUBLE,
                  landmarks.data(), s_per_rank * dim, MPI_DOUBLE, MPI_COMM_WORLD);

    // 3. Compute W_local (Local points vs Global landmarks)
    MatrixXd W_local(local_n, s_total);
    double sigma2 = sigma_ * sigma_;

    #pragma omp parallel for schedule(static)
    for(int i=0; i<local_n; ++i) {
        // Cache-friendly blocking over landmarks
        for (int j_blk = 0; j_blk < s_total; j_blk += 64) {
            int j_lim = std::min(j_blk + 64, s_total);
            for (int j = j_blk; j < j_lim; ++j) {
                double d2 = (localData.row(i) - landmarks.row(j)).squaredNorm();
                W_local(i, j) = std::exp(-d2 / sigma2);
            }
        }
    }

    // 4. Eigen-decomposition on Rank 0
    MatrixXd projection(s_total, k_);
    if(rank == 0) {
        MatrixXd W_s(s_total, s_total);
        for(int i=0; i<s_total; ++i)
            for(int j=0; j<s_total; ++j)
                W_s(i, j) = std::exp(-(landmarks.row(i) - landmarks.row(j)).squaredNorm() / sigma2);
        
        SelfAdjointEigenSolver<MatrixXd> es(W_s);
        VectorXd inv_sqrt_L = es.eigenvalues().tail(k_).array().abs().inverse().sqrt();
        projection = es.eigenvectors().rightCols(k_) * inv_sqrt_L.asDiagonal();
    }
    MPI_Bcast(projection.data(), s_total * k_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 5. Project to spectral space and L2-normalize rows
    MatrixXd local_U = W_local * projection;
    W_local.resize(0,0); // Clear memory

    #pragma omp parallel for schedule(static)
    for(int i=0; i<local_n; ++i) {
        double row_n = std::sqrt(local_U.row(i).squaredNorm());
        if(row_n > 1e-12) local_U.row(i) /= row_n;
    }

    // Get rows_count for Gatherv later
    std::vector<int> rows_count(size);
    MPI_Allgather(&local_n, 1, MPI_INT, rows_count.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // 6. Distributed K-means
    kmeans_hpc(local_U, N_global, rows_count);
}

void SpectralClusteringMPI::kmeans_hpc(const MatrixXd& localX, int N_global, const std::vector<int>& rows_count) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int local_n = (int)localX.rows();
    int dim = (int)localX.cols();
    int n_threads = omp_get_max_threads();

    MatrixXd centers(k_, dim);
    if(rank == 0) {
        std::mt19937 gen(777);
        for(int i=0; i<k_; ++i) centers.row(i) = localX.row(gen() % local_n);
    }
    MPI_Bcast(centers.data(), k_ * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Pre-allocate thread buffers to avoid allocations in loop
    std::vector<MatrixXd> thread_sums(n_threads, MatrixXd::Zero(k_, dim));
    std::vector<VectorXi> thread_counts(n_threads, VectorXi::Zero(k_));
    std::vector<int> local_labels(local_n);

    for(int iter=0; iter<100; ++iter) {
        for(int t=0; t<n_threads; ++t) {
            thread_sums[t].setZero();
            thread_counts[t].setZero();
        }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(static)
            for(int i=0; i<local_n; ++i) {
                int best_c = 0;
                double min_d = (localX.row(i) - centers.row(0)).squaredNorm();
                for(int c=1; c<k_; ++c) {
                    double d = (localX.row(i) - centers.row(c)).squaredNorm();
                    if(d < min_d) { min_d = d; best_c = c; }
                }
                local_labels[i] = best_c;
                thread_sums[tid].row(best_c) += localX.row(i);
                thread_counts[tid](best_c)++;
            }
        }

        MatrixXd local_sum_all = MatrixXd::Zero(k_, dim);
        VectorXi local_count_all = VectorXi::Zero(k_);
        for(int t=0; t<n_threads; ++t) {
            local_sum_all += thread_sums[t];
            local_count_all += thread_counts[t];
        }

        MatrixXd global_sums(k_, dim);
        VectorXi global_counts(k_);
        MPI_Allreduce(local_sum_all.data(), global_sums.data(), k_*dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_count_all.data(), global_counts.data(), k_, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        for(int c=0; c<k_; ++c)
            if(global_counts(c) > 0) centers.row(c) = global_sums.row(c) / (double)global_counts(c);
    }

    // Final result gathering on Rank 0
    if(rank == 0) labels_.resize(N_global);
    std::vector<int> displs(size, 0);
    for(int r=1; r<size; ++r) displs[r] = displs[r-1] + rows_count[r-1];

    MPI_Gatherv(local_labels.data(), local_n, MPI_INT, 
                rank == 0 ? labels_.data() : nullptr, 
                rows_count.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
}