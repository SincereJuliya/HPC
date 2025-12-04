#include "SpectralClustering.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <sstream>

/**
 * @brief Reads CSV with header x,y
 */
Eigen::MatrixXd read_csv(const std::string& filename) {
    std::ifstream file(filename);
    if(!file.is_open()){
        std::cerr << "Cannot open file: " << filename << "\n";
        return Eigen::MatrixXd(0,0);
    }

    std::string line;
    std::getline(file, line); // skip header
    std::vector<Eigen::RowVector2d> points;
    while(std::getline(file,line)){
        double x,y;
        std::stringstream ss(line);
        std::string token;
        if(std::getline(ss, token, ',')) x = std::stod(token);
        if(std::getline(ss, token, ',')) y = std::stod(token);
        points.push_back({x,y});
    }

    Eigen::MatrixXd data(points.size(),2);
    for(size_t i=0;i<points.size();++i)
        data.row(i)=points[i];
    return data;
}

/**
 * @brief Saves labels to CSV
 */
void save_labels_csv(const std::string& filename, const std::vector<int>& labels){
    std::filesystem::path p(filename);
    if(!p.parent_path().empty())
        std::filesystem::create_directories(p.parent_path());

    std::ofstream file(filename);
    if(!file.is_open()){
        std::cerr << "Cannot write to: " << filename << "\n";
        return;
    }

    file << "index,label\n";
    for(size_t i=0;i<labels.size();++i)
        file << i << "," << labels[i] << "\n";

    std::cout << "Labels saved to " << filename << std::endl;
}

int main(int argc, char** argv){
    std::string filename = "../scripts/data/mixed_dataset.csv";
    int k = 8;             // default clusters
    int knn = 7;           // default k-NN
    double sigma = 0.04;   // default sigma
    int kmeans_runs = 4;   // default K-means runs

    // Parse command line arguments: filename k knn sigma kmeans_runs
    if(argc > 1) filename = argv[1];
    if(argc > 2) k = std::stoi(argv[2]);
    if(argc > 3) knn = std::stoi(argv[3]);
    if(argc > 4) sigma = std::stod(argv[4]);
    if(argc > 5) kmeans_runs = std::stoi(argv[5]);

    Eigen::MatrixXd data = read_csv(filename);
    if(data.rows()==0 || data.cols()!=2){
        std::cerr << "Data not loaded or wrong format.\n";
        return 1;
    }

    std::cout << "Read " << data.rows() << " points.\n";
    std::cout << "Running spectral clustering with k=" << k
              << ", knn=" << knn
              << ", sigma=" << sigma
              << ", kmeans_runs=" << kmeans_runs << std::endl;

    SpectralClustering sc(k, knn, sigma, kmeans_runs);
    sc.fit(data);

    auto labels = sc.get_labels();
    save_labels_csv("../data/mixed_dataset_labels.csv", labels);

    return 0;
}
