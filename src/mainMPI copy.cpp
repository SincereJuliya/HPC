#include "SpectralClusteringMPI.hpp"
#include <Eigen/Dense>
#include <mpi.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

Eigen::MatrixXd read_csv_root(const std::string &fname){
    std::ifstream f(fname);
    if(!f.is_open()) return Eigen::MatrixXd(0,0);
    std::string line;
    std::getline(f,line); // skip header
    std::vector<Eigen::RowVector2d> rows;
    while(std::getline(f,line)){
        double a,b;
        if(sscanf(line.c_str(), "%lf,%lf", &a, &b) == 2){
            rows.push_back({a,b});
        }
    }
    Eigen::MatrixXd M(rows.size(), 2);
    for(size_t i=0;i<rows.size(); ++i) M.row(i) = rows[i];
    return M;
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Default parameters (assumes execution from project root)
    std::string datafile = "data/mixed_dataset.csv"; 
    int k = 8;           
    int knn = 10;        
    double sigma = 0.05; 
    int runs = 10;       

    if(argc > 1) datafile = argv[1];
    if(argc > 2) k = std::atoi(argv[2]);
    if(argc > 3) knn = std::atoi(argv[3]);
    if(argc > 4) sigma = std::atof(argv[4]);
    if(argc > 5) runs = std::atoi(argv[5]);

    Eigen::MatrixXd data;
    if(rank == 0){
        std::cout << "=== Hybrid MPI+OpenMP Spectral Clustering ===" << std::endl;
        std::cout << "Dataset: " << datafile << std::endl;
        std::cout << "Params: K=" << k << ", Knn=" << knn << ", Sigma=" << sigma << ", Runs=" << runs << std::endl;

        data = read_csv_root(datafile);
        if(data.rows() == 0){
            std::cerr << "Error: Root cannot open or empty file " << datafile << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int N = 0, D = 0;
    if(rank == 0){ N = (int)data.rows(); D = (int)data.cols(); }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(N == 0 || D == 0){
        MPI_Finalize();
        return 1;
    }

    std::vector<double> buffer;
    if(rank == 0){
        buffer.resize((size_t)N * (size_t)D);
        for(int i=0;i<N;++i)
            for(int j=0;j<D;++j)
                buffer[(size_t)i * D + j] = data(i,j);
    } else {
        buffer.resize((size_t)N * (size_t)D);
    }
    MPI_Bcast(buffer.data(), N*D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixRowMajor;
    MatrixRowMajor data_mat = MatrixRowMajor::Zero(N, D);
    std::memcpy(data_mat.data(), buffer.data(), sizeof(double) * (size_t)N * (size_t)D);

    // Execution
    SpectralClusteringMPI sc(k, knn, sigma, runs);
    sc.fit(data_mat);

    if(rank == 0){
        auto labels = sc.get_labels();
        std::string outfile = "data/mixed_dataset_labels_mpi.csv";
        std::ofstream out(outfile);
        if(out.is_open()){
            out << "index,label\n";
            for(size_t i=0;i<labels.size(); ++i) out << i << "," << labels[i] << "\n";
            out.close();
            std::cout << "Results saved to: " << outfile << "\n";
        } else {
            std::cerr << "Error: Could not write to " << outfile << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
