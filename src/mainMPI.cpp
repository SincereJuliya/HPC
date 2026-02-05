#include "SpectralClusteringMPI.hpp"
#include <Eigen/Dense>
#include <mpi.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>

Eigen::MatrixXd read_csv_root(const std::string &fname) {
    std::ifstream f(fname);
    if(!f.is_open()) return Eigen::MatrixXd(0,0);
    std::string line;
    std::getline(f,line); // skip header
    std::vector<Eigen::RowVector2d> rows;
    while(std::getline(f,line)) {
        double a,b;
        if(sscanf(line.c_str(), "%lf,%lf", &a, &b) == 2)
            rows.push_back({a,b});
    }
    Eigen::MatrixXd M(rows.size(), 2);
    for(size_t i=0; i<rows.size(); ++i) M.row(i) = rows[i];
    return M;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- START TIMER ---
    double start_time = MPI_Wtime();

    std::string datafile = "data/mixed_dataset.csv";
    int k=8, knn=10, runs=5;
    double sigma=0.05;

    if(argc > 1) datafile = argv[1];
    if(argc > 2) k = std::atoi(argv[2]);
    if(argc > 3) knn = std::atoi(argv[3]);
    if(argc > 4) sigma = std::atof(argv[4]);
    if(argc > 5) runs = std::atoi(argv[5]);

    int N_global = 0, D = 2;
    std::vector<double> full_buffer;

    if(rank == 0) {
        Eigen::MatrixXd full_data = read_csv_root(datafile);
        N_global = (int)full_data.rows();
        D = (int)full_data.cols();
        full_buffer.resize((size_t)N_global * D);
        Eigen::Map<MatrixRowMajor>(full_buffer.data(), N_global, D) = full_data;
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Dataset: " << datafile << std::endl;
        std::cout << "Global N: " << N_global << " | Ranks: " << size << std::endl;
    }

    MPI_Bcast(&N_global, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> sendcounts(size), displs(size);
    int rem = N_global % size;
    int offset = 0;
    for (int i = 0; i < size; i++) {
        int rows = (N_global / size) + (i < rem ? 1 : 0);
        sendcounts[i] = rows * D;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int local_n = (N_global / size) + (rank < rem ? 1 : 0);
    MatrixRowMajor local_data(local_n, D);

    MPI_Scatterv(rank == 0 ? full_buffer.data() : nullptr, sendcounts.data(), displs.data(), 
                 MPI_DOUBLE, local_data.data(), local_n * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    SpectralClusteringMPI sc(k, knn, sigma, runs);
    sc.fit(local_data, N_global);

    // --- END TIMER ---
    double end_time = MPI_Wtime();

    if(rank == 0) {
        auto labels = sc.get_labels();
        std::ofstream out("results/labels.csv");
        out << "index,label\n";
        for(size_t i=0; i<labels.size(); ++i) out << i << "," << labels[i] << "\n";
        
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Total execution time: " << (end_time - start_time) << " s" << std::endl;
        std::cout << "Clustering complete." << std::endl;
        std::cout << "========================================\n" << std::endl;
    }

    MPI_Finalize();
    return 0;
}