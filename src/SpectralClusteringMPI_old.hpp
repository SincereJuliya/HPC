#ifndef EC8AEC96_3DED_4F60_83C9_2A13F378B05F
#define EC8AEC96_3DED_4F60_83C9_2A13F378B05F
#include <Eigen/Dense>
#include <vector>
#include <mpi.h>

class SpectralClusteringMPI {
public:
    SpectralClusteringMPI(int k=3, int knn=10, double sigma=-1.0, int kmeans_runs=5);

    // data: full NxD matrix (same on all ranks or broadcasted to all)
    void fit(const Eigen::MatrixXd& data);

    // valid only on rank 0 after fit()
    const std::vector<int>& get_labels() const { return labels_; }

private:
    int k_;
    int knn_;
    double sigma_;
    int kmeans_runs_;
    std::vector<int> labels_; // populated on rank 0

    Eigen::MatrixXd standardize(const Eigen::MatrixXd& X);
    double estimate_sigma(const Eigen::MatrixXd& X);

    // compute similarity weights for local block: Xblock rows x Ncols
    Eigen::MatrixXd compute_similarity_block(const Eigen::MatrixXd& Xblock, const Eigen::MatrixXd& Xfull);

    Eigen::MatrixXd compute_laplacian(const Eigen::MatrixXd& W);

    // distributed K-means: localX is local block of eigenvectors (row-major mapping is fine)
    double kmeans_mpi(const Eigen::MatrixXd& localX, int global_rows, const std::vector<int>& rows_count, int n_iter=200, int seed_offset=0);
};


#endif /* EC8AEC96_3DED_4F60_83C9_2A13F378B05F */
