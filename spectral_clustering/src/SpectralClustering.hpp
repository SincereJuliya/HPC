#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>

/**
 * @class SpectralClustering
 * @brief Spectral clustering with k-NN graph and normalized Laplacian.
 *
 * Features:
 * - Manual or automatic sigma
 * - Standardization of input
 * - k-NN similarity graph
 * - Normalized Laplacian L_sym
 * - Row-normalized eigenvectors before K-means
 * - Multiple K-means runs for better initialization
 */
class SpectralClustering {
public:
    /**
     * @brief Constructor
     * @param k Number of clusters
     * @param knn Number of nearest neighbors
     * @param sigma RBF width (-1 = auto estimate)
     * @param kmeans_runs Number of K-means initializations
     */
    SpectralClustering(int k = 3, int knn = 10, double sigma = -1.0, int kmeans_runs = 5);

    /**
     * @brief Fit clustering
     * @param data NxD data matrix
     */
    void fit(const Eigen::MatrixXd& data);

    /**
     * @brief Get cluster labels
     * @return labels
     */
    const std::vector<int>& get_labels() const { return labels_; }

private:
    int k_;
    int knn_;
    double sigma_;
    int kmeans_runs_;
    std::vector<int> labels_;

    Eigen::MatrixXd compute_similarity(const Eigen::MatrixXd& X);
    Eigen::MatrixXd compute_laplacian(const Eigen::MatrixXd& W);
    Eigen::MatrixXd standardize(const Eigen::MatrixXd& X);
    double estimate_sigma(const Eigen::MatrixXd& X);
    void kmeans(const Eigen::MatrixXd& X, int n_iter = 100);
};
