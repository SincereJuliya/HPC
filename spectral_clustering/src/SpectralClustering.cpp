#include "SpectralClustering.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <numeric>
#include <vector>
#include <random>

using namespace Eigen;

// ---------- Constructor ----------
SpectralClustering::SpectralClustering(int k, int knn, double sigma, int kmeans_runs)
    : k_(k), knn_(knn), sigma_(sigma), kmeans_runs_(kmeans_runs) {}

// ---------- Standardize ----------
MatrixXd SpectralClustering::standardize(const MatrixXd& X) {
    MatrixXd Xs = X;
    RowVectorXd mean = X.colwise().mean();
    RowVectorXd stddev = ((X.rowwise() - mean).array().square().colwise().sum() / (X.rows()-1)).sqrt();
    for(int j=0;j<X.cols();++j)
        if(stddev(j) > 1e-8) Xs.col(j) = (Xs.col(j).array() - mean(j)) / stddev(j);
    return Xs;
}

// ---------- Estimate global sigma ----------
double SpectralClustering::estimate_sigma(const MatrixXd& X) {
    std::vector<double> dists;
    for(int i=0;i<X.rows();++i)
        for(int j=i+1;j<X.rows();++j)
            dists.push_back((X.row(i)-X.row(j)).norm());
    std::nth_element(dists.begin(), dists.begin()+dists.size()/2, dists.end());
    double med = dists[dists.size()/2];
    return med * 0.2; // уменьшенный σ для сложного датасета
}

// ---------- Compute similarity with local scaling ----------
MatrixXd SpectralClustering::compute_similarity(const MatrixXd& X) {
    int N = X.rows();
    MatrixXd W = MatrixXd::Zero(N,N);

    if(sigma_ < 0) sigma_ = estimate_sigma(X);
    std::cout << "Using global sigma = " << sigma_ << std::endl;

    // Precompute distances to all points
    std::vector<std::vector<std::pair<double,int>>> all_dists(N);
    for(int i=0;i<N;++i){
        for(int j=0;j<N;++j){
            if(i==j) continue;
            double d = (X.row(i)-X.row(j)).norm();
            all_dists[i].push_back({d,j});
        }
        std::sort(all_dists[i].begin(), all_dists[i].end(),
                  [](const auto &a, const auto &b){ return a.first<b.first; });
    }

    // Compute local σ_i = distance to k-th nearest neighbor
    std::vector<double> local_sigma(N);
    for(int i=0;i<N;++i)
        local_sigma[i] = all_dists[i][knn_-1].first;

    // Build weighted k-NN graph with local scaling
    for(int i=0;i<N;++i){
        for(int n=0;n<knn_;++n){
            int j = all_dists[i][n].second;
            double sigma_ij = std::sqrt(local_sigma[i]*local_sigma[j]);
            double w = std::exp(-std::pow(all_dists[i][n].first,2)/(2*sigma_ij*sigma_ij));
            W(i,j) = w;
            W(j,i) = w;
        }
        if(i%100==0) std::cout << "Processed row " << i << "/" << N << "\r";
    }
    std::cout << std::endl;
    return W;
}

// ---------- Laplacian ----------
MatrixXd SpectralClustering::compute_laplacian(const MatrixXd& W) {
    int N = W.rows();
    VectorXd D = W.rowwise().sum();
    MatrixXd D_inv_sqrt = D.array().inverse().sqrt().matrix().asDiagonal();
    return MatrixXd::Identity(N,N) - D_inv_sqrt * W * D_inv_sqrt;
}

// ---------- K-means with multiple runs ----------
void SpectralClustering::kmeans(const MatrixXd& X, int n_iter) {
    int N = X.rows();
    int dim = X.cols();
    labels_.resize(N,0);

    std::mt19937 rng(std::random_device{}());
    MatrixXd best_centers;
    std::vector<int> best_labels(N);
    double best_inertia = std::numeric_limits<double>::max();

    for(int run=0; run<kmeans_runs_; ++run){
        // random initialization
        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        MatrixXd centers(k_, dim);
        for(int j=0;j<k_;++j) centers.row(j) = X.row(indices[j]);

        std::vector<int> labels(N,0);

        for(int it=0; it<n_iter; ++it){
            // assign
            for(int i=0;i<N;++i){
                double best = (X.row(i)-centers.row(0)).squaredNorm();
                int best_idx = 0;
                for(int j=1;j<k_;++j){
                    double d = (X.row(i)-centers.row(j)).squaredNorm();
                    if(d<best){ best = d; best_idx=j; }
                }
                labels[i] = best_idx;
            }
            // update
            MatrixXd new_centers = MatrixXd::Zero(k_,dim);
            VectorXi counts = VectorXi::Zero(k_);
            for(int i=0;i<N;++i){
                new_centers.row(labels[i]) += X.row(i);
                counts(labels[i]) += 1;
            }
            for(int j=0;j<k_;++j)
                if(counts(j)>0) new_centers.row(j)/=counts(j);
            centers = new_centers;
        }

        // compute inertia
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

// ---------- Fit ----------
void SpectralClustering::fit(const MatrixXd& data) {
    std::cout << "Standardizing data..." << std::endl;
    MatrixXd Xs = standardize(data);

    std::cout << "Computing similarity..." << std::endl;
    MatrixXd W = compute_similarity(Xs);

    std::cout << "Computing Laplacian..." << std::endl;
    MatrixXd L = compute_laplacian(W);

    std::cout << "Computing eigenvectors..." << std::endl;
    SelfAdjointEigenSolver<MatrixXd> es(L);

    int n_eig = std::min(2*k_, (int)L.rows());
    MatrixXd eigvecs = es.eigenvectors().leftCols(n_eig);

    // Row-normalize eigenvectors
    for(int i=0;i<eigvecs.rows();++i){
        double norm = eigvecs.row(i).norm();
        if(norm>1e-8) eigvecs.row(i)/=norm;
    }

    std::cout << "Running K-means on eigenvectors..." << std::endl;
    kmeans(eigvecs, 200);
    std::cout << "Done clustering!" << std::endl;
}
