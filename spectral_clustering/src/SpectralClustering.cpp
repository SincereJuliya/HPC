#include "SpectralClustering.hpp"
#include <cmath>
#include <limits>
#include <random>
#include <iostream>

SpectralClustering::SpectralClustering(int k, double sigma)
    : k_(k), sigma_(sigma) {}

std::vector<int> SpectralClustering::get_labels() const {
    return labels_;
}

void SpectralClustering::fit(const Eigen::MatrixXd& data) {
    if(data.rows() == 0 || data.cols() == 0){
        std::cerr << "Empty dataset!\n";
        return;
    }
    compute_similarity(data);
    compute_laplacian();
    compute_eigenvectors();
    kmeans(U_);
}

void SpectralClustering::compute_similarity(const Eigen::MatrixXd& data) {
    int N = data.rows();
    W_ = Eigen::MatrixXd::Zero(N, N);

    for(int i = 0; i < N; ++i){
        for(int j = i; j < N; ++j){
            double dist = (data.row(i) - data.row(j)).squaredNorm();
            double val = std::exp(-dist / (2*sigma_*sigma_));
            W_(i,j) = val;
            W_(j,i) = val;
        }
    }
}

void SpectralClustering::compute_laplacian() {
    Eigen::VectorXd D = W_.rowwise().sum();
    L_ = Eigen::MatrixXd::Zero(W_.rows(), W_.cols());
    for(int i = 0; i < W_.rows(); ++i){
        L_(i,i) = D(i);
    }
    L_ -= W_;
}

void SpectralClustering::compute_eigenvectors() {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(L_);
    if(solver.info() != Eigen::Success){
        std::cerr << "Eigen decomposition failed!\n";
        return;
    }
    U_ = solver.eigenvectors().leftCols(k_);
}

void SpectralClustering::kmeans(const Eigen::MatrixXd& data, int iterations) {
    int N = data.rows();
    int D = data.cols();
    labels_ = std::vector<int>(N, 0);

    std::vector<Eigen::VectorXd> centroids(k_, Eigen::VectorXd(D));
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, N-1);
    for(int i=0;i<k_;++i) centroids[i] = data.row(dis(gen));

    for(int iter=0; iter<iterations; ++iter){
        // assignment
        for(int i=0;i<N;++i){
            double best = std::numeric_limits<double>::max();
            int bi = 0;
            for(int j=0;j<k_;++j){
                double dist = (data.row(i) - centroids[j].transpose()).squaredNorm();
                if(dist<best){ best=dist; bi=j; }
            }
            labels_[i]=bi;
        }

        // update
        std::vector<Eigen::VectorXd> new_centroids(k_, Eigen::VectorXd::Zero(D));
        std::vector<int> counts(k_,0);
        for(int i=0;i<N;++i){
            new_centroids[labels_[i]] += data.row(i);
            counts[labels_[i]] += 1;
        }
        for(int j=0;j<k_;++j){
            if(counts[j]>0) new_centroids[j] /= counts[j];
            centroids[j] = new_centroids[j];
        }
    }
}
