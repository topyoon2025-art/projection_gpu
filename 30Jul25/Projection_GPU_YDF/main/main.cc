#include "projection.h"
#include "absl/status/status.h"
#include "absl/log/log.h"  // If you're using LOG(INFO)
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include <google/protobuf/repeated_field.h>
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"


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
#include <chrono>

int main (int argc, char* argv[]) {
  absl::Status gpu_status = CheckHasGPU(true);

  if (!gpu_status.ok()) {
    std::cerr << "GPU Check failed: " << gpu_status.message() << std::endl;
    return 1;
  }

  std::cout << "GPU check passed.\n";
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num_examples> <num_features>\n";
        return 1;
    }
    std::cout << std::setprecision(4);
    // const int num_examples = 11; //Rows
    // const int num_features = 10; //Columns (features)

    // Step 1: Load dataset
    const std::string path = "csv:./random_data.csv"; 
    std::cout << "Testing 0:\n";
    yggdrasil_decision_forests::dataset::VerticalDataset train_dataset;
    yggdrasil_decision_forests::dataset::proto::DataSpecification data_spec;
    yggdrasil_decision_forests::dataset::LoadConfig config;
    yggdrasil_decision_forests::dataset::proto::DataSpecificationGuide guide;
    const bool use_flume = false;
    absl::Status spec_status = yggdrasil_decision_forests::dataset::CreateDataSpecWithStatus(path, use_flume, guide, &data_spec);

    //std::optional<std::vector<int>> required_columns = std::vector<int>{0, 2, 4};

    absl::Status status = yggdrasil_decision_forests::dataset::LoadVerticalDataset(path, data_spec, &train_dataset, {}, config);
    if (!status.ok()) {
    std::cerr << "Failed to load dataset: " << status.message() << std::endl;
    return 1;
    }
    std::cout << "Number of rows: " << train_dataset.nrow() << std::endl;
    std::cout << "Number of columns: " << train_dataset.ncol() << std::endl;
  
    // Step 2: Define numerical feature indices
    google::protobuf::RepeatedField<int32_t> numerical_features;
    numerical_features.Add(0);  // Example feature indices
    numerical_features.Add(1);
    numerical_features.Add(2);
    std::cout << "Numerical Features\n" << std::endl;
      for (float value : numerical_features) {
        std::cout << value << std::endl;
    }
    std::cout << std::endl;

    // Step 3: Create ProjectionEvaluator
    yggdrasil_decision_forests::model::decision_tree::internal::ProjectionEvaluator evaluator(train_dataset, numerical_features);

    // Step 4: Define a Projection
    yggdrasil_decision_forests::model::decision_tree::internal::Projection projection;
    projection.push_back({.attribute_idx = 0, .weight = 1.0f});
    projection.push_back({.attribute_idx = 1, .weight = -0.5f});
    projection.push_back({.attribute_idx = 2, .weight = 0.25f});

    // Step 5: Select example indices
    using yggdrasil_decision_forests::dataset::UnsignedExampleIdx;
    std::vector<UnsignedExampleIdx> selected_indices;
    for (UnsignedExampleIdx i = 0; i < train_dataset.nrow(); ++i) {
        selected_indices.push_back(i);
    }
    absl::Span<const UnsignedExampleIdx> selected_examples(selected_indices);

    // Step 6: Evaluate projection
    std::vector<float> values;
    absl::Status eval_status = evaluator.Evaluate(projection, absl::MakeSpan(selected_examples), &values);
    if (!eval_status.ok()) {
        std::cerr << "Evaluation failed: " << eval_status.message() << std::endl;
        return 1;
    }

    // Step 7: Print results
    for (size_t i = 0; i < values.size(); ++i) {
        std::cout << "Example " << selected_examples[i] << ": " << values[i] << std::endl;
    }

  
    const int num_examples = std::atoi(argv[1]);
    const int num_features = std::atoi(argv[2]);

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
    //std::cout << "Randomly generated dataset:\n";
    for (int i = 0; i < num_examples; ++i) {
        for (int j = 0; j < num_features; ++j) {
            dataset[i][j] = test_dist(gen);
            //std::cout << dataset[i][j] << "\t";
        }   
        //std::cout << std::endl;  
    }
    //std::cout << std::endl; 

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

    //GPU
    //std::cout << "Beginning in GPU.\n" << std::endl;
    //enter chrono
    std::vector<float> projected_gpu(row_numbers.size());
    
    absl::Status status_gpu = cudaFlatProjection(
        absl::MakeConstSpan(flat_dataset),
        absl::MakeConstSpan(row_numbers),
        absl::MakeConstSpan(column_numbers),
        num_features,
        absl::MakeSpan(projected_gpu));
    
    //std::cout << "Ending in GPU.\n" << std::endl;
    std::cout << "First value from GPU computation: "<< projected_gpu[0] << std::endl;
    std::cout << std::endl; 

    if (!status_gpu.ok()) {
    std::cerr << "CUDA FlatProjection failed: " << status_gpu.message() << std::endl;
    return 1;
    }

    //CPU
    //std::cout << "Beginning in CPU.\n" << std::endl;
    //enter chrono
    std::vector<float> projected_cpu(row_numbers.size());
    auto startA = std::chrono::high_resolution_clock::now();
    absl::Status status_cpu = FlatProjection(
        absl::MakeConstSpan(flat_dataset),
        absl::MakeConstSpan(row_numbers),
        absl::MakeConstSpan(column_numbers),
        num_features,
        absl::MakeSpan(projected_cpu));

    auto endA = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> durationA = endA - startA;
    std::cout << "CPU elapsed time: " << durationA.count() << " ms\n";
    //std::cout << "Ending in CPU.\n" << std::endl;
      // for (float value : combined_flat_projected_dataset) {
    //     std::cout << value << std::endl;
    // }
    // std::cout << std::endl;
    std::cout << "First value from CPU computation: " << projected_cpu[0] << std::endl;
    std::cout << std::endl; 

    return 0;
}