add_executable(test_assignment test_assignment.cpp)
target_link_libraries(test_assignment PRIVATE Catch2::Catch2WithMain CUDA::cudart kmeans)