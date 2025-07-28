#include "projection_gpu.h"

#include <stdio.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include <cstdlib>   
#include <ctime>    
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>

std::vector<float> FlatProjection(const std::vector<float>& dataset, const std::vector<int>& row_indices, const std::vector<int>& column_indices, int num_features) {
    std::vector<float> bootstrapped_dataset(row_indices.size() * column_indices.size());
    std::vector<float> projected_dataset(row_indices.size());
    std::cout << "Combined dataset Flat in one function:" << std::endl;
    for (long unsigned int i = 0; i < row_indices.size(); i++) {
        for (long unsigned int j = 0; j < column_indices.size(); j++) {
            bootstrapped_dataset[i * column_indices.size() + j] = dataset[row_indices[i] * num_features + column_indices[j]];
            projected_dataset[i] += bootstrapped_dataset[i * column_indices.size() + j];
            std::cout << bootstrapped_dataset[i * column_indices.size() + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl; 
    return projected_dataset;
}

int main() {
    std::cout << std::setprecision(4);
    const int num_examples = 11; //Rows
    const int num_features = 10; //Columns (features)
    const int total_size = num_examples * num_features; //Total size of elemenets row * column
    const double rows_bootstrap_percentage = .8; // 80% of rows
    const int bootstrapped_columns_count = 3; // 3 columns

    //Random number
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> test_dist(0.0f, 10.0f);

    // Generate Dataset
    std::vector<std::vector<float>> dataset(num_examples, std::vector<float>(num_features));
    std::cout << std::endl; 
    std::cout << "Randomly generated dataset:\n";
    for (int i = 0; i < num_examples; ++i) {
        for (int j = 0; j < num_features; ++j) {
            dataset[i][j] = test_dist(gen);
            std::cout << dataset[i][j] << "\t";
        }   
        std::cout << std::endl;  
    }
    std::cout << std::endl; 

    //Flatten dataset for CUDA
    std::cout << "Dataset total size:  " << total_size << std::endl;
    std::vector<float> flat_dataset(total_size);
    for (int i = 0; i < num_examples; ++i) {
        for (int j = 0; j < num_features; ++j) {
            flat_dataset[i * num_features + j] = dataset[i][j];
        }
    }
    std::cout << std::endl; 

    //Rounded number of 80% rows and select the rows
    int bootstrapped_rows_count = round(num_examples * rows_bootstrap_percentage);
    std::vector<int> row_numbers(num_examples);
    std::iota(row_numbers.begin(), row_numbers.end(), 0);
    std::shuffle(row_numbers.begin(), row_numbers.end(), gen);
    row_numbers.resize(bootstrapped_rows_count);

    //Print randomly chosen row numbers
    std::cout << "Randomly chosen row numbers: ";
    for (int num : row_numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl; 
  
    //Fixed number of 3 columns
    std::vector<int> column_numbers(num_features); 
    std::iota(column_numbers.begin(), column_numbers.end(), 0);
    std::shuffle(column_numbers.begin(), column_numbers.end(), gen);
    column_numbers.resize(bootstrapped_columns_count);

    //Print randomly chosen column numbers
    std::cout << "Randomly chosen column numbers: ";
    for (int num : column_numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl; 

    std::vector<float> combined_flat_projected_dataset = FlatProjection(flat_dataset, row_numbers, column_numbers, num_features);
    std::cout << "Projecting in one function CPU:\n";
    for (float value : combined_flat_projected_dataset) {
        std::cout << value << std::endl;
    }
    std::cout << std::endl;

    std::vector<float> projected = cudaFlatProjection(flat_dataset, row_numbers, column_numbers, num_features);
    std::cout << "Ending in GPU.\n" << std::endl;

    return 0;
}