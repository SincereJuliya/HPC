#pragma once
#include <Eigen/Dense>
#include <vector>
#include <mpi.h>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixRowMajor;

class SpectralClusteringMPI {
public:
    SpectralClusteringMPI(int k=3, int knn=10, double sigma=-1.0, int kmeans_runs=5);

    void fit(Eigen::Ref<MatrixRowMajor> localData, int N_global);

    const std::vector<int>& get_labels() const { return labels_; }

private:
    int k_;
    int knn_;
    double sigma_;
    int kmeans_runs_;
    std::vector<int> labels_;

    void standardize_distributed(Eigen::Ref<MatrixRowMajor> localData, int N_global);
    void kmeans_hpc(const Eigen::MatrixXd& localX, int N_global, const std::vector<int>& rows_count);
};