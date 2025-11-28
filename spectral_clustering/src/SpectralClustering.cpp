#include "SpectralClustering.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <numeric>
#include <vector>
#include <random>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;
using namespace Eigen;

// ---------- Constructor ----------
SpectralClustering::SpectralClustering(int k, int knn, double sigma, int kmeans_runs)
    : k_(k), knn_(knn), sigma_(sigma), kmeans_runs_(kmeans_runs) {}

// ---------- Standardize ----------
MatrixXd SpectralClustering::standardize(const MatrixXd& X) {
    RowVectorXd mean = X.colwise().mean();
    RowVectorXd stddev = ((X.rowwise() - mean).array().square().colwise().sum() / (X.rows()-1)).sqrt();
    RowVectorXd std_safe = stddev.unaryExpr([](double x){ return x < 1e-8 ? 1.0 : x; });
    return (X.rowwise() - mean).array().rowwise() / std_safe.array();
}

// ---------- Estimate global sigma ----------
double SpectralClustering::estimate_sigma(const MatrixXd& X) {
    std::vector<double> dists;
    int N = X.rows();
    dists.reserve(N*(N-1)/2);
    for(int i=0;i<N;++i)
        for(int j=i+1;j<N;++j)
            dists.push_back((X.row(i)-X.row(j)).norm());

    std::nth_element(dists.begin(), dists.begin() + dists.size()/2, dists.end());
    double med = dists[dists.size()/2];
    return med * 0.2;
}

// ---------- Dense similarity ----------
MatrixXd SpectralClustering::compute_similarity_dense(const MatrixXd& X) {
    int N = X.rows();
    MatrixXd W = MatrixXd::Zero(N,N);

    if(sigma_ < 0) sigma_ = estimate_sigma(X);
    //std::cout << "Using global sigma = " << sigma_ << std::endl;

    std::vector<std::vector<std::pair<double,int>>> all_dists(N);
    for(int i=0;i<N;++i){
        for(int j=0;j<N;++j){
            if(i==j) continue;
            double d = (X.row(i)-X.row(j)).norm();
            all_dists[i].push_back({d,j});
        }
        std::nth_element(all_dists[i].begin(), all_dists[i].begin()+knn_, all_dists[i].end(),
                         [](auto &a, auto &b){ return a.first < b.first; });
        std::sort(all_dists[i].begin(), all_dists[i].begin()+knn_,
                  [](auto &a, auto &b){ return a.first < b.first; });
    }

    std::vector<double> local_sigma(N);
    for(int i=0;i<N;++i)
        local_sigma[i] = all_dists[i][knn_-1].first;

    for(int i=0;i<N;++i){
        for(int n=0;n<knn_;++n){
            int j = all_dists[i][n].second;
            double sigma_ij = std::sqrt(local_sigma[i]*local_sigma[j]);
            double w = std::exp(-std::pow(all_dists[i][n].first,2)/(2*sigma_ij*sigma_ij));
            W(i,j) = w;
            W(j,i) = w;
        }
        //if(i%100==0) std::cerr << "Processed row " << i << "/" << N << "\r";
    }
    std::cerr << std::endl;
    return W;
}

// ---------- Sparse similarity ----------
SparseMatrix<double> SpectralClustering::compute_similarity_sparse(const MatrixXd& X) {
    int N = X.rows();
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(N*knn_*2);

    if(sigma_ < 0) sigma_ = estimate_sigma(X);
    //std::cout << "Using global sigma = " << sigma_ << std::endl;

    std::vector<std::vector<std::pair<double,int>>> all_dists(N);
    for(int i=0;i<N;++i){
        for(int j=0;j<N;++j){
            if(i==j) continue;
            double d = (X.row(i)-X.row(j)).norm();
            all_dists[i].push_back({d,j});
        }
        std::nth_element(all_dists[i].begin(), all_dists[i].begin()+knn_, all_dists[i].end(),
                         [](auto &a, auto &b){ return a.first < b.first; });
        std::sort(all_dists[i].begin(), all_dists[i].begin()+knn_,
                  [](auto &a, auto &b){ return a.first < b.first; });
    }

    std::vector<double> local_sigma(N);
    for(int i=0;i<N;++i)
        local_sigma[i] = all_dists[i][knn_-1].first;

    for(int i=0;i<N;++i){
        for(int n=0;n<knn_;++n){
            int j = all_dists[i][n].second;
            double sigma_ij = std::sqrt(local_sigma[i]*local_sigma[j]);
            double w = std::exp(-std::pow(all_dists[i][n].first,2)/(2*sigma_ij*sigma_ij));
            triplets.emplace_back(i,j,w);
            triplets.emplace_back(j,i,w);
        }
        if(i%100==0) std::cerr << "Processed row " << i << "/" << N << "\r";
    }
    std::cerr << std::endl;

    SparseMatrix<double> W(N,N);
    W.setFromTriplets(triplets.begin(), triplets.end());
    return W;
}

// ---------- Laplacian ----------
MatrixXd SpectralClustering::compute_laplacian(const MatrixXd& W) {
    VectorXd D = W.rowwise().sum();
    MatrixXd D_inv_sqrt = D.array().inverse().sqrt().matrix().asDiagonal();
    return MatrixXd::Identity(W.rows(),W.cols()) - D_inv_sqrt * W * D_inv_sqrt;
}

// ---------- K-means ----------
void SpectralClustering::kmeans(const MatrixXd& X, int n_iter) {
    int N = X.rows();
    int dim = X.cols();
    labels_.resize(N,0);

    std::mt19937 rng(std::random_device{}());
    MatrixXd best_centers;
    std::vector<int> best_labels(N);
    double best_inertia = std::numeric_limits<double>::max();

    for(int run=0; run<kmeans_runs_; ++run){
        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        MatrixXd centers(k_, dim);
        for(int j=0;j<k_;++j) centers.row(j) = X.row(indices[j]);

        std::vector<int> labels(N,0);

        for(int it=0; it<n_iter; ++it){
            for(int i=0;i<N;++i){
                double best = (X.row(i)-centers.row(0)).squaredNorm();
                int best_idx = 0;
                for(int j=1;j<k_;++j){
                    double d = (X.row(i)-centers.row(j)).squaredNorm();
                    if(d<best){ best = d; best_idx=j; }
                }
                labels[i] = best_idx;
            }

            MatrixXd new_centers = MatrixXd::Zero(k_,dim);
            VectorXi counts = VectorXi::Zero(k_);
            for(int i=0;i<N;++i){
                new_centers.row(labels[i]) += X.row(i);
                counts(labels[i]) += 1;
            }
            for(int j=0;j<k_;++j)
                if(counts(j)>0) new_centers.row(j)/=counts(j);

            if((centers - new_centers).norm() < 1e-6) break;
            centers = new_centers;
        }

        double inertia = 0.0;
        for(int i=0;i<N;++i)
            inertia += (X.row(i)-centers.row(labels[i])).squaredNorm();

        if(inertia < best_inertia){
            best_inertia = inertia;
            best_centers = centers;
            best_labels = labels;
        }
    }

    labels_ = best_labels;
}

void SpectralClustering::fit(const MatrixXd& data) {

    auto t_start_total = Clock::now();

    // ---------- Standardize ----------
    std::cout << "[1/5] Standardize: ";
    auto t1 = Clock::now();
    MatrixXd Xs = standardize(data);
    auto t2 = Clock::now();
    std::cout << std::chrono::duration<double>(t2 - t1).count()
              << " s\n";

    // ---------- Similarity ----------
    std::cout << "[2/5] Local similarity block: ";
    t1 = Clock::now();
    MatrixXd W = compute_similarity_dense(Xs);
    t2 = Clock::now();
    std::cout << std::chrono::duration<double>(t2 - t1).count()
              << " s\n";

    // ---------- Laplacian ----------
    std::cout << "[3/5] Laplacian: ";
    t1 = Clock::now();
    MatrixXd L = compute_laplacian(W);
    t2 = Clock::now();
    std::cout << std::chrono::duration<double>(t2 - t1).count()
              << " s\n";

    // ---------- Eigen decomposition ----------
    std::cout << "[4/5] Eigenvectors: ";
    t1 = Clock::now();
    SelfAdjointEigenSolver<MatrixXd> es(L);
    t2 = Clock::now();
    std::cout << std::chrono::duration<double>(t2 - t1).count()
              << " s\n";

    int n_eig = std::min(2*k_, (int)L.rows());
    MatrixXd eigvecs = es.eigenvectors().leftCols(n_eig);

    // normalize rows
    for(int i = 0; i < eigvecs.rows(); ++i){
        double norm = eigvecs.row(i).norm();
        if(norm > 1e-8) eigvecs.row(i) /= norm;
    }

    // ---------- K-means ----------
    std::cout << "[5/5] K-means: ";
    t1 = Clock::now();
    kmeans(eigvecs, 200);
    t2 = Clock::now();
    std::cout << std::chrono::duration<double>(t2 - t1).count()
              << " s\n";

    auto t_end_total = Clock::now();
    std::cout << "--------------------------------------\n";
    std::cout << "Total Spectral Clustering time: "
              << std::chrono::duration<double>(t_end_total - t_start_total).count()
              << " s\n";
    std::cout << "--------------------------------------\n";

    std::cout << "Done clustering!" << std::endl;
}

