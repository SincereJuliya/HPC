#include "SpectralClustering.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

Eigen::MatrixXd read_csv(const std::string& filename){
    std::ifstream file(filename);
    if(!file.is_open()){
        std::cerr << "Cannot open file: " << filename << "\n";
        return Eigen::MatrixXd(0,0);
    }

    std::string line;
    std::vector<std::vector<double>> data;
    if(!getline(file, line)) {
        std::cerr << "File is empty or missing header\n";
        return Eigen::MatrixXd(0,0);
    }

    while(getline(file, line)){
        std::stringstream ss(line);
        std::string val;
        std::vector<double> row;
        while(getline(ss,val,',')){
            try {
                row.push_back(std::stod(val));
            } catch(...) {
                std::cerr << "Invalid number: " << val << "\n";
                return Eigen::MatrixXd(0,0);
            }
        }
        if(row.size() < 2) continue;
        data.push_back({row[0], row[1]});
    }

    if(data.empty()){
        std::cerr << "No valid rows found in CSV!\n";
        return Eigen::MatrixXd(0,0);
    }

    Eigen::MatrixXd mat(data.size(), 2);
    for(int i=0;i<data.size();++i){
        mat(i,0)=data[i][0];
        mat(i,1)=data[i][1];
    }

    std::cout << "Read " << data.size() << " points.\n";
    return mat;
}

// Save labels to CSV
void save_labels_csv(const std::string& filename, const std::vector<int>& labels) {
    std::ofstream file(filename);
    if(!file.is_open()){
        std::cerr << "Cannot open file for writing: " << filename << "\n";
        return;
    }

    file << "index,label\n";
    for(size_t i=0; i<labels.size(); ++i){
        file << i << "," << labels[i] << "\n";
    }
    std::cout << "Labels saved to " << filename << "\n";
}


int main(int argc, char** argv){
    std::string filename = "../scripts/data/mixed_dataset.csv";

    Eigen::MatrixXd data = read_csv(filename);
    if(data.rows() == 0){
        std::cerr << "No data loaded. Exiting.\n";
        return 1;
    } else {
        std::cerr << "Data loaded successfully. Rows: " << data.rows() << "\n";
    }

    int k = 8;
    double sigma = 1.0;

    SpectralClustering sc(k, sigma);
    sc.fit(data);

    auto labels = sc.get_labels();

    save_labels_csv("../data/mixed_dataset_labels.csv", labels);

    std::cout << "Saved." << "\n";

    return 0;
}
