#include "SpectralClustering.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>

/**
 * @brief Reads CSV with header x,y
 */
Eigen::MatrixXd read_csv(const std::string& filename) {
    std::ifstream file(filename);
    if(!file.is_open()){ std::cerr << "Cannot open file: " << filename << "\n"; return Eigen::MatrixXd(0,0); }

    std::string line;
    std::getline(file, line); // skip header
    std::vector<Eigen::Vector2d> points;
    while(std::getline(file,line)){
        double x,y;
        if(sscanf(line.c_str(),"%lf,%lf",&x,&y)==2)
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
    std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
    std::ofstream file(filename);
    if(!file.is_open()){ std::cerr << "Cannot write to: " << filename << "\n"; return; }
    file << "index,label\n";
    for(size_t i=0;i<labels.size();++i) file << i << "," << labels[i] << "\n";
    std::cout << "Labels saved to " << filename << std::endl;
}

int main(int argc, char** argv){
    std::string filename = "../scripts/data/mixed_dataset.csv";
    Eigen::MatrixXd data = read_csv(filename);
    if(data.rows()==0){ std::cerr << "No data loaded.\n"; return 1; }
    std::cout << "Read " << data.rows() << " points." << std::endl;

    int k = 6;        // clusters
    int knn = 7;      // k-NN
    double sigma = 0.04; // 
    int kmeans_runs = 4;  // K-means runs

    SpectralClustering sc(k, knn, sigma, kmeans_runs);
    sc.fit(data);

    auto labels = sc.get_labels();
    save_labels_csv("../data/mixed_dataset_labels.csv", labels);

    return 0;
}
