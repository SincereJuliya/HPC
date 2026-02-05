#include "SpectralClusteringMPI.hpp"
#include <Eigen/Eigenvalues>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>
#include <chrono>
#include <omp.h>

using Clock = std::chrono::high_resolution_clock;
using namespace Eigen;

SpectralClusteringMPI::SpectralClusteringMPI(int k,int knn,double sigma,int kmeans_runs)
    : k_(k), knn_(knn), sigma_(sigma), kmeans_runs_(kmeans_runs) { }

MatrixXd SpectralClusteringMPI::standardize(const MatrixXd& X){
    RowVectorXd mean = X.colwise().mean();
    RowVectorXd stddev = ((X.rowwise()-mean).array().square().colwise().sum()/(X.rows()-1)).sqrt();
    RowVectorXd safe = stddev.unaryExpr([](double v){ return v < 1e-8 ? 1.0 : v; });
    return (X.rowwise() - mean).array().rowwise() / safe.array();
}

double SpectralClusteringMPI::estimate_sigma(const MatrixXd& X){ return 1.0; }

// --- ИЗМЕНЁННЫЙ compute_similarity_block (kNN-sparse) ---
MatrixXd SpectralClusteringMPI::compute_similarity_block(const MatrixXd& Xblock, const MatrixXd& Xfull){
    int local_n = (int)Xblock.rows();
    int N = (int)Xfull.rows();
    int dim = (int)Xblock.cols();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(local_n == 0 || N == 0) return MatrixXd(0,0);

    // --- ИЗМЕНЕНО: используем только K ближайших соседей ---
    struct Neighbor { int idx; double w; };
    std::vector<std::vector<Neighbor>> sparse_W(local_n);

    #pragma omp parallel for schedule(static)
    for(int il=0; il<local_n; ++il){
        std::vector<std::pair<double,int>> dist_idx;
        for(int j=0;j<N;++j){
            double d = (Xblock.row(il)-Xfull.row(j)).squaredNorm();
            dist_idx.push_back({d,j});
        }
        int k_neighbor = std::min(knn_, N-1);
        std::nth_element(dist_idx.begin(), dist_idx.begin()+k_neighbor, dist_idx.end());
        sparse_W[il].resize(k_neighbor);
        for(int ni=0; ni<k_neighbor; ++ni){
            sparse_W[il][ni] = { dist_idx[ni].second, std::exp(-dist_idx[ni].first/(sigma_*sigma_)) };
        }
    }

    // --- Временно возвращаем dense Wblock для совместимости ---
    MatrixXd Wblock(local_n, N);
    Wblock.setZero();
    for(int i=0;i<local_n;++i)
        for(auto &nb : sparse_W[i]) Wblock(i, nb.idx) = nb.w;

    return Wblock;
}

// Normalized Symmetric Laplacian: I - D^-1/2 * W * D^-1/2
MatrixXd SpectralClusteringMPI::compute_laplacian(const MatrixXd& W){
    VectorXd D = W.rowwise().sum();
    for(int i=0;i<D.size();++i) if(D(i) <= 0) D(i) = 1e-12;
    MatrixXd D_inv_sqrt = D.array().inverse().sqrt().matrix().asDiagonal();
    return MatrixXd::Identity(W.rows(), W.cols()) - D_inv_sqrt * W * D_inv_sqrt;
}

// Distributed K-Means
double SpectralClusteringMPI::kmeans_mpi(const MatrixXd& localX, int global_rows, const std::vector<int>& rows_count, int n_iter, int seed_offset){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int local_n = (int)localX.rows();
    int dim = (int)localX.cols();

    MatrixXd centers = MatrixXd::Zero(k_, dim);
    MatrixXd local_init = MatrixXd::Zero(k_, dim);
    std::mt19937_64 rng((unsigned)(std::chrono::system_clock::now().time_since_epoch().count() + rank + seed_offset));

    if(local_n > 0){
        std::uniform_int_distribution<int> ud(0, local_n - 1);
        for(int c=0;c<k_;++c){
            int idx = ud(rng);
            local_init.row(c) = localX.row(idx);
        }
    }
    MPI_Allreduce(local_init.data(), centers.data(), k_*dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    centers /= (double)size;

    std::vector<int> local_labels(local_n, 0);
    double current_sse = 0.0;

    for(int iter=0; iter<n_iter; ++iter){
        current_sse = 0.0;
        #pragma omp parallel for reduction(+:current_sse)
        for(int i=0;i<local_n;++i){
            double best = (localX.row(i) - centers.row(0)).squaredNorm();
            int best_idx = 0;
            for(int c=1;c<k_;++c){
                double d = (localX.row(i) - centers.row(c)).squaredNorm();
                if(d < best){ best = d; best_idx = c; }
            }
            local_labels[i] = best_idx;
            current_sse += best;
        }
        MatrixXd local_sum = MatrixXd::Zero(k_, dim);
        VectorXi local_count = VectorXi::Zero(k_);
        for(int i=0;i<local_n;++i){
            local_sum.row(local_labels[i]) += localX.row(i);
            local_count(local_labels[i]) += 1;
        }
        MatrixXd global_sum = MatrixXd::Zero(k_, dim);
        VectorXi global_count = VectorXi::Zero(k_);
        MPI_Allreduce(local_sum.data(), global_sum.data(), k_*dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_count.data(), global_count.data(), k_, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        for(int c=0;c<k_;++c)
            if(global_count(c) > 0) centers.row(c) = global_sum.row(c) / global_count(c);
    }

    double global_sse = 0.0;
    MPI_Allreduce(&current_sse, &global_sse, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    std::vector<int> sendcounts(size), displs(size);
    int offset = 0;
    for(int r=0;r<size;++r){ sendcounts[r] = rows_count[r]; displs[r] = offset; offset += sendcounts[r]; }
    if(rank == 0) labels_.assign(global_rows, 0);
    MPI_Gatherv(local_labels.data(), local_n, MPI_INT, rank == 0 ? labels_.data() : nullptr,
                sendcounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    return global_sse;
}

// --- ИЗМЕНЁННЫЙ fit() с Nyström ---
void SpectralClusteringMPI::fit(const MatrixXd& data){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto t1 = Clock::now();
    MatrixXd Xs = standardize(data);
    int N = (int)Xs.rows();
    int dim = (int)Xs.cols();

    int rows_per_proc = N / size;
    int remainder = N % size;
    std::vector<int> rows_count(size);
    for(int r=0;r<size;++r) rows_count[r] = rows_per_proc + (r < remainder ? 1 : 0);
    int row_start = 0;
    for(int r=0;r<rank;++r) row_start += rows_count[r];
    int local_n = rows_count[rank];

    if(sigma_ < 0 && rank == 0) sigma_ = 1.0;
    MPI_Bcast(&sigma_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MatrixXd Xblock = (local_n > 0) ? Xs.block(row_start, 0, local_n, dim) : MatrixXd(0, dim);

    // --- Nyström sampling ---
    int s = std::min(2000, N);
    std::vector<int> sample_idx(s);
    if(rank==0){
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dis(0, N-1);
        for(int i=0;i<s;++i) sample_idx[i] = dis(gen);
    }
    MPI_Bcast(sample_idx.data(), s, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local similarity to sampled points
    MatrixXd W_local(local_n, s);
    for(int i=0;i<local_n;++i){
        for(int j=0;j<s;++j){
            double dist = (Xblock.row(i)-Xs.row(sample_idx[j])).squaredNorm();
            W_local(i,j) = std::exp(-dist/(sigma_*sigma_));
        }
    }

    // Reduce to root only
    MatrixXd W_sample;
    if(rank==0) W_sample = MatrixXd(s,s);
    MPI_Reduce(rank==0 ? MPI_IN_PLACE : W_local.data(),
               rank==0 ? W_sample.data() : W_local.data(),
               local_n*s, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Eigen decomposition on root
    MatrixXd eigvecs_sample;
    if(rank==0){
        SelfAdjointEigenSolver<MatrixXd> es(W_sample);
        eigvecs_sample = es.eigenvectors().rightCols(k_);
    }

    MPI_Bcast(eigvecs_sample.data(), s*k_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Approx eigenvectors for local points
    MatrixXd eigvecs_block(local_n, k_);
    for(int i=0;i<local_n;++i){
        for(int j=0;j<k_;++j){
            double sum = 0;
            for(int l=0;l<s;++l) sum += W_local(i,l) * eigvecs_sample(l,j);
            eigvecs_block(i,j) = sum;
        }
    }

    typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixRowMajor;
    MatrixRowMajor localEig(local_n, k_);
    std::memcpy(localEig.data(), eigvecs_block.data(), sizeof(double)*local_n*k_);

    // --- K-means ---
    if(rank==0) std::cout << "[Step 2/3] Eigen Decomposition Done. Starting Distributed K-Means..." << std::endl;
    t1 = Clock::now();
    double best_sse = 1e30;
    std::vector<int> best_labels;
    if(rank==0) best_labels.assign(N, 0);
    int runs = (kmeans_runs_ < 1) ? 1 : kmeans_runs_;

    for(int r=0; r<runs; ++r) {
        double sse = kmeans_mpi(localEig, N, rows_count, 200, r * 100);
        if(rank == 0) {
            if(sse < best_sse) { best_sse = sse; best_labels = labels_; }
        }
    }

    if(rank == 0) labels_ = best_labels;
    auto t2 = Clock::now();
    if(rank==0) {
        std::cout << "[Step 3/3] K-means (Best of " << runs << " runs): "
                  << std::chrono::duration<double>(t2 - t1).count() << " s\n";
    }
}