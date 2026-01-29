#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>

/**
 * @class SpectralClustering
 * @brief Sequential implementation of Spectral Clustering (Baseline).
 *
 * Features:
 * - Self-Tuning Similarity (Zelnik-Manor & Perona)
 * - Normalized Symmetric Laplacian
 * - Standard K-Means
 */
class SpectralClustering {
public:
    SpectralClustering(int k = 3, int knn = 10, double sigma = -1.0, int kmeans_runs = 5);

    void fit(const Eigen::MatrixXd& data);

    const std::vector<int>& get_labels() const { return labels_; }

private:
    int k_;
    int knn_;
    double sigma_;
    int kmeans_runs_;
    std::vector<int> labels_;

    Eigen::MatrixXd compute_similarity_dense(const Eigen::MatrixXd& X);
    Eigen::MatrixXd compute_laplacian(const Eigen::MatrixXd& W);
    Eigen::MatrixXd standardize(const Eigen::MatrixXd& X);
    double estimate_sigma(const Eigen::MatrixXd& X);
    void kmeans(const Eigen::MatrixXd& X, int n_iter = 100);
};
