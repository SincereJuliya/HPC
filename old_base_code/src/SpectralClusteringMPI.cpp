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
    // NON PIU' USATO con Zelnik-Manor, ma lasciato per compatibilità
    return 1.0;
}

MatrixXd SpectralClusteringMPI::compute_similarity_block(const MatrixXd& Xblock, const MatrixXd& Xfull){
    int local_n = (int)Xblock.rows();
    int N = (int)Xfull.rows();
    MatrixXd Wblock = MatrixXd::Zero(local_n, N);

    // Serve per MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(local_n == 0 && N == 0) return Wblock;

    // 1. Calcolo distanze locali -> globali
    // Memorizziamo tutte le distanze perché servono sia per sigma che per W
    // Questo usa memoria (Local_N * N), ma è necessario per l'efficienza
    std::vector<std::vector<double>> dists_matrix(local_n);
    std::vector<double> local_sigmas(local_n);

    for(int il=0; il<local_n; ++il){
        dists_matrix[il].resize(N);
        std::vector<double> row_dists;
        row_dists.reserve(N);

        for(int j=0;j<N;++j){
            double d = (Xblock.row(il) - Xfull.row(j)).norm();
            dists_matrix[il][j] = d;
            row_dists.push_back(d);
        }

        // 2. Trova il K-esimo vicino per determinare sigma_i (Zelnik-Manor)
        // Usiamo knn_ come valore K (consigliato 7)
        int k_neighbor = std::min(knn_, N-1);
        if (k_neighbor < 1) k_neighbor = 1;

        std::nth_element(row_dists.begin(), row_dists.begin()+k_neighbor, row_dists.end());
        local_sigmas[il] = row_dists[k_neighbor];

        // Evita divisioni per zero
        if(local_sigmas[il] < 1e-6) local_sigmas[il] = 1e-6;
    }

    // 3. MPI: Condividere i sigma locali con TUTTI (Allgatherv)
    // Dobbiamo ricostruire counts e displs per sapere quanti sigma arrivano da chi
    std::vector<int> counts(size), displs(size);
    int rows_per_proc = N / size;
    int remainder = N % size;
    int off = 0;
    for(int r=0; r<size; ++r){
        counts[r] = rows_per_proc + (r < remainder ? 1 : 0);
        displs[r] = off;
        off += counts[r];
    }

    std::vector<double> global_sigmas(N);
    MPI_Allgatherv(local_sigmas.data(), local_n, MPI_DOUBLE,
                   global_sigmas.data(), counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    // 4. Calcolo Pesi con Self-Tuning: W_ij = exp( -d^2 / (sigma_i * sigma_j) )
    for(int il=0; il<local_n; ++il){
        double sig_i = local_sigmas[il];
        // Recuperiamo l'indice globale della riga corrente per sapere chi siamo (opzionale per diagonale)
        // int global_i = displs[rank] + il;

        for(int j=0; j<N; ++j){
            // Salta se stesso (opzionale, exp(0)=1, ma nei grafi spesso W_ii=0)
            // if (global_i == j) { Wblock(il, j) = 0.0; continue; }

            double d = dists_matrix[il][j];
            double sig_j = global_sigmas[j];

            // Formula Zelnik-Manor & Perona
            double w = std::exp(- (d*d) / (sig_i * sig_j));

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

double SpectralClusteringMPI::kmeans_mpi(const MatrixXd& localX, int global_rows, const std::vector<int>& rows_count, int n_iter, int seed_offset){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = (int)localX.rows();
    int dim = (int)localX.cols();

    // Init casuale migliore
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
        // assignment
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

    return global_sse;
}

void SpectralClusteringMPI::fit(const MatrixXd& data){
    auto t_total_start = Clock::now();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // [1/6] Standardize
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

    // NOTA: estimate_sigma non serve più perché calcoliamo i sigma locali in compute_similarity_block
    // ma lasciamo la chiamata MPI_Bcast per non rompere la struttura se qualcuno la usa
    if(sigma_ < 0 && rank == 0) sigma_ = 1.0;
    MPI_Bcast(&sigma_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MatrixXd Xblock = (local_n > 0) ? Xs.block(row_start, 0, local_n, dim) : MatrixXd(0, dim);

    // [2/6] Similarity Block (Zelnik-Manor Self-Tuning)
    auto t_sim_start = Clock::now();
    MatrixXd Wblock = compute_similarity_block(Xblock, Xs);
    auto t_sim_end = Clock::now();
    if(rank==0) std::cout << "[2/6] Self-Tuning Similarity: " << std::chrono::duration<double>(t_sim_end - t_sim_start).count() << " s\n";

    // Gather W
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
    MPI_Gatherv(sendbuf_W.data(), sendbuf_W.size(), MPI_DOUBLE,
                rank == 0 ? recvbuf_W.data() : nullptr,
                sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int n_eig = 0;
    std::vector<double> sendbuf_eig;

    // [4/6] Eigenvectors
    if(rank == 0){
        MatrixXd W(N, N);
        for(int i=0;i<N;++i)
            for(int j=0;j<N;++j)
                W(i, j) = recvbuf_W[i * N + j];

        MatrixXd L = compute_laplacian(W);
        SelfAdjointEigenSolver<MatrixXd> es(L);

        n_eig = k_; // Usiamo k autovettori esatti

        MatrixXd eigvecs = es.eigenvectors().leftCols(n_eig);
        for(int i=0;i<N;++i){
            double norm = eigvecs.row(i).norm();
            if(norm > 1e-8) eigvecs.row(i) /= norm;
        }

        sendbuf_eig.resize((size_t)N * (size_t)n_eig);
        for(int i=0;i<N;++i)
            for(int j=0;j<n_eig;++j)
                sendbuf_eig[(size_t)i * n_eig + j] = eigvecs(i, j);
    }

    MPI_Bcast(&n_eig, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(n_eig <= 0) return;

    // Scatter Eigenvectors
    std::vector<int> sendcounts_elems(size), displs_elems(size);
    int off2 = 0;
    for(int r=0;r<size;++r){
        sendcounts_elems[r] = rows_count[r] * n_eig;
        displs_elems[r] = off2;
        off2 += sendcounts_elems[r];
    }
    int local_elems = rows_count[rank] * n_eig;
    std::vector<double> recvbuf_local_eig(local_elems);

    MPI_Scatterv(rank == 0 ? sendbuf_eig.data() : nullptr,
                 sendcounts_elems.data(), displs_elems.data(), MPI_DOUBLE,
                 recvbuf_local_eig.data(), local_elems, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixRowMajor;
    MatrixRowMajor localEig(rows_count[rank], n_eig);
    if(local_elems > 0)
        std::memcpy(localEig.data(), recvbuf_local_eig.data(), sizeof(double) * local_elems);

    // [6/6] K-means con RETRY (Runs)
    t1 = Clock::now();

    double best_sse = 1e30;
    std::vector<int> best_labels;
    if(rank==0) best_labels.assign(N, 0);

    int runs = (kmeans_runs_ < 1) ? 1 : kmeans_runs_;

    for(int r=0; r<runs; ++r) {
        double sse = kmeans_mpi(localEig, N, rows_count, 200, r * 100);
        if(rank == 0) {
            std::cout << " Run " << r+1 << "/" << runs << " SSE: " << sse << std::endl;
            if(sse < best_sse) {
                best_sse = sse;
                best_labels = labels_;
            }
        }
    }

    if(rank == 0) labels_ = best_labels;

    auto t2 = Clock::now();
    if(rank==0)
        std::cout << "[6/6] K-means (Best of " << runs << "): "
                  << std::chrono::duration<double>(t2 - t1).count() << " s\n";

    if(rank==0) std::cout << "Done.\n";
}