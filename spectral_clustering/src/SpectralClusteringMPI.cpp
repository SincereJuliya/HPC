#include "SpectralClusteringMPI.hpp"
#include <Eigen/Eigenvalues>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>
#include <cassert>
#include <cstring>
#include <chrono>

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

double SpectralClusteringMPI::estimate_sigma(const MatrixXd& X){
    int N = (int)X.rows();
    if(N < 2) return 1.0;
    std::vector<double> dists;
    dists.reserve((size_t)N*(N-1)/2);
    for(int i=0;i<N;++i)
        for(int j=i+1;j<N;++j)
            dists.push_back((X.row(i)-X.row(j)).norm());
    std::nth_element(dists.begin(), dists.begin()+dists.size()/2, dists.end());
    return dists[dists.size()/2] * 0.2;
}

MatrixXd SpectralClusteringMPI::compute_similarity_block(const MatrixXd& Xblock, const MatrixXd& Xfull){
    int local_n = (int)Xblock.rows();
    int N = (int)Xfull.rows();
    MatrixXd Wblock = MatrixXd::Zero(local_n, N);

    if(local_n == 0) return Wblock;

    // compute distances to all points
    std::vector<std::vector<std::pair<double,int>>> all_dists(local_n);
    for(int il=0; il<local_n; ++il){
        all_dists[il].reserve(N);
        for(int j=0;j<N;++j){
            double d = (Xblock.row(il) - Xfull.row(j)).norm();
            all_dists[il].push_back({d,j});
        }
        int k = std::min(knn_, N-1);
        std::nth_element(all_dists[il].begin(), all_dists[il].begin()+k, all_dists[il].end(),
                         [](const auto &a,const auto &b){ return a.first < b.first; });
        std::sort(all_dists[il].begin(), all_dists[il].begin()+k,
                  [](const auto &a,const auto &b){ return a.first < b.first; });
    }

    // local sigma estimate per local row
    std::vector<double> local_sigma(local_n);
    for(int il=0; il<local_n; ++il){
        int k = std::min(knn_, N-1);
        local_sigma[il] = all_dists[il][k-1].first;
        if(local_sigma[il] <= 0) local_sigma[il] = 1e-8;
    }

    // build Wblock using simplified local scaling (sigma_j approximated by sigma_i to avoid extra communication)
    for(int il=0; il<local_n; ++il){
        int k = std::min(knn_, N-1);
        for(int n=0;n<k;++n){
            int j = all_dists[il][n].second;
            double d = all_dists[il][n].first;
            double sigma_ij = std::sqrt(local_sigma[il] * local_sigma[il]); // simplified
            double w = std::exp(- (d*d) / (2.0 * sigma_ij * sigma_ij));
            Wblock(il, j) = w;
        }
    }

    return Wblock;
}

MatrixXd SpectralClusteringMPI::compute_laplacian(const MatrixXd& W){
    VectorXd D = W.rowwise().sum();
    for(int i=0;i<D.size();++i) if(D(i) <= 0) D(i) = 1e-12;
    MatrixXd D_inv_sqrt = D.array().inverse().sqrt().matrix().asDiagonal();
    return MatrixXd::Identity(W.rows(), W.cols()) - D_inv_sqrt * W * D_inv_sqrt;
}

void SpectralClusteringMPI::kmeans_mpi(const MatrixXd& localX, int global_rows, const std::vector<int>& rows_count, int n_iter){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = (int)localX.rows();
    int dim = (int)localX.cols();
    // initialize centers: root will create an initial center set by sampling global (we do Allreduce average of random picks)
    MatrixXd centers = MatrixXd::Zero(k_, dim);
    MatrixXd local_init = MatrixXd::Zero(k_, dim);
    std::mt19937_64 rng((unsigned)(std::random_device{}() + rank));
    if(local_n > 0){
        std::uniform_int_distribution<int> ud(0, local_n - 1);
        for(int c=0;c<k_;++c){
            int idx = ud(rng);
            local_init.row(c) = localX.row(idx);
        }
    } else {
        // leave local_init zeros
    }
    // average local_init to get global centers (simple init)
    MPI_Allreduce(local_init.data(), centers.data(), k_*dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    centers /= (double)size;

    std::vector<int> local_labels(local_n, 0);

    for(int iter=0; iter<n_iter; ++iter){
        // assignment
        for(int i=0;i<local_n;++i){
            double best = (localX.row(i) - centers.row(0)).squaredNorm();
            int best_idx = 0;
            for(int c=1;c<k_;++c){
                double d = (localX.row(i) - centers.row(c)).squaredNorm();
                if(d < best){ best = d; best_idx = c; }
            }
            local_labels[i] = best_idx;
        }

        // local sums & counts
        MatrixXd local_sum = MatrixXd::Zero(k_, dim);
        VectorXi local_count = VectorXi::Zero(k_);
        for(int i=0;i<local_n;++i){
            local_sum.row(local_labels[i]) += localX.row(i);
            local_count(local_labels[i]) += 1;
        }

        // global reduce
        MatrixXd global_sum = MatrixXd::Zero(k_, dim);
        VectorXi global_count = VectorXi::Zero(k_);
        MPI_Allreduce(local_sum.data(), global_sum.data(), k_*dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_count.data(), global_count.data(), k_, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // update centers
        for(int c=0;c<k_;++c)
            if(global_count(c) > 0) centers.row(c) = global_sum.row(c) / global_count(c);
    }

    // gather labels to rank 0 with Gatherv using rows_count
    std::vector<int> sendcounts(size), displs(size);
    int offset = 0;
    for(int r=0;r<size;++r){
        sendcounts[r] = rows_count[r];
        displs[r] = offset;
        offset += sendcounts[r];
    }

    if(rank == 0) labels_.assign(global_rows, 0);

    MPI_Gatherv(local_labels.data(), local_n, MPI_INT,
                rank == 0 ? labels_.data() : nullptr,
                sendcounts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
}

void SpectralClusteringMPI::fit(const MatrixXd& data){
    auto t_total_start = Clock::now();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---------- Standardize ----------
    auto t1 = Clock::now();
    MatrixXd Xs = standardize(data);
    auto t2 = Clock::now();
    if(rank==0)
        std::cout << "[1/6] Standardize: "
                  << std::chrono::duration<double>(t2 - t1).count()
                  << " s\n";

    int N = (int)Xs.rows();
    int dim = (int)Xs.cols();

    // partition
    int rows_per_proc = N / size;
    int remainder = N % size;
    std::vector<int> rows_count(size);
    for(int r=0;r<size;++r)
        rows_count[r] = rows_per_proc + (r < remainder ? 1 : 0);

    int row_start = 0;
    for(int r=0;r<rank;++r) row_start += rows_count[r];
    int local_n = rows_count[rank];

    // sigma
    if(sigma_ < 0 && rank == 0) sigma_ = estimate_sigma(Xs);
    MPI_Bcast(&sigma_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MatrixXd Xblock =
        (local_n > 0) ? Xs.block(row_start, 0, local_n, dim)
                       : MatrixXd(0, dim);

    // ---------- Local similarity block ----------
    t1 = Clock::now();
    MatrixXd Wblock = compute_similarity_block(Xblock, Xs);
    t2 = Clock::now();
    if(rank==0)
        std::cout << "[2/6] Local similarity block: "
                  << std::chrono::duration<double>(t2 - t1).count()
                  << " s\n";

    // pack
    std::vector<double> sendbuf_W(local_n * N);
    for(int i=0;i<local_n;++i)
        for(int j=0;j<N;++j)
            sendbuf_W[i * N + j] = Wblock(i, j);

    std::vector<int> sendcounts(size), displs(size);
    int offset = 0;
    for(int r=0;r<size;++r){
        sendcounts[r] = rows_count[r] * N;
        displs[r] = offset;
        offset += sendcounts[r];
    }

    std::vector<double> recvbuf_W;
    if(rank == 0) recvbuf_W.assign((size_t)N * (size_t)N, 0.0);

    // ---------- Gatherv ----------
    t1 = Clock::now();
    MPI_Gatherv(sendbuf_W.data(), sendbuf_W.size(), MPI_DOUBLE,
                rank == 0 ? recvbuf_W.data() : nullptr,
                sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    t2 = Clock::now();
    if(rank==0)
        std::cout << "[3/6] MPI_Gatherv(W): "
                  << std::chrono::duration<double>(t2 - t1).count()
                  << " s\n";

    int n_eig = 0;
    std::vector<double> sendbuf_eig;

    // ---------- Root: Laplacian + Eigenvectors ----------
    if(rank == 0){
        auto t_root1 = Clock::now();

        MatrixXd W(N, N);
        for(int i=0;i<N;++i)
            for(int j=0;j<N;++j)
                W(i, j) = recvbuf_W[i * N + j];

        MatrixXd L = compute_laplacian(W);

        SelfAdjointEigenSolver<MatrixXd> es(L);
        n_eig = std::min(2 * k_, N);
        MatrixXd eigvecs = es.eigenvectors().leftCols(n_eig);

        for(int i=0;i<N;++i){
            double norm = eigvecs.row(i).norm();
            if(norm > 1e-8) eigvecs.row(i) /= norm;
        }

        sendbuf_eig.resize((size_t)N * (size_t)n_eig);
        for(int i=0;i<N;++i)
            for(int j=0;j<n_eig;++j)
                sendbuf_eig[(size_t)i * n_eig + j] = eigvecs(i, j);

        auto t_root2 = Clock::now();
        std::cout << "[4/6] Laplacian + Eigenvectors: "
                  << std::chrono::duration<double>(t_root2 - t_root1).count()
                  << " s\n";
    }

    MPI_Bcast(&n_eig, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(n_eig <= 0) return;

    // scatter
    std::vector<int> sendcounts_elems(size), displs_elems(size);
    int off2 = 0;
    for(int r=0;r<size;++r){
        sendcounts_elems[r] = rows_count[r] * n_eig;
        displs_elems[r] = off2;
        off2 += sendcounts_elems[r];
    }

    int local_elems = rows_count[rank] * n_eig;
    std::vector<double> recvbuf_local_eig(local_elems);

    // ---------- Scatterv eigenvectors ----------
    t1 = Clock::now();
    MPI_Scatterv(rank == 0 ? sendbuf_eig.data() : nullptr,
                 sendcounts_elems.data(), displs_elems.data(), MPI_DOUBLE,
                 recvbuf_local_eig.data(), local_elems, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    t2 = Clock::now();
    if(rank==0)
        std::cout << "[5/6] MPI_Scatterv(eigvecs): "
                  << std::chrono::duration<double>(t2 - t1).count()
                  << " s\n";

    typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixRowMajor;
    MatrixRowMajor localEig(rows_count[rank], n_eig);
    if(local_elems > 0)
        std::memcpy(localEig.data(), recvbuf_local_eig.data(),
                    sizeof(double) * local_elems);

    // ---------- K-means ----------
    t1 = Clock::now();
    kmeans_mpi(localEig, N, rows_count, 200);
    t2 = Clock::now();
    if(rank==0)
        std::cout << "[6/6] K-means: "
                  << std::chrono::duration<double>(t2 - t1).count()
                  << " s\n";

    // ---------- Total ----------
    auto t_total_end = Clock::now();
    if(rank==0)
        std::cout << "----------------------------------------\n"
                  << "Total MPI Spectral Clustering time: "
                  << std::chrono::duration<double>(t_total_end - t_total_start).count()
                  << " s\n"
                  << "----------------------------------------\n";
}

