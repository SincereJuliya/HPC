#pragma once
#include <vector>
#include <Eigen/Dense>

/**
 * @brief Spectral Clustering class
 */
class SpectralClustering {
public:
    SpectralClustering(int k, double sigma);

    /**
     * @brief Fit the model to the dataset
     * @param data NxD matrix (N points, D dimensions)
     */
    void fit(const Eigen::MatrixXd& data);

    /**
     * @brief Get cluster labels after fitting
     */
    std::vector<int> get_labels() const;

private:
    int k_;             // number of clusters
    double sigma_;      // for RBF kernel
    Eigen::MatrixXd W_; // similarity matrix
    Eigen::MatrixXd L_; // Laplacian
    Eigen::MatrixXd U_; // matrix of eigenvectors
    std::vector<int> labels_;

    void compute_similarity(const Eigen::MatrixXd& data);
    void compute_laplacian();
    void compute_eigenvectors();
    void kmeans(const Eigen::MatrixXd& data, int iterations = 100);
};
