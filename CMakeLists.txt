cmake_minimum_required(VERSION 3.13)

project(cse6230-final-project CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fopenmp -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp")

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 70 72)
set(CMAKE_CUDA_FLAGS_DEBUG "-G")
set(CMAKE_CUDA_FLAG_RELEASE "-Wno-deprecated-gpu-targets")

find_package(CUDAToolkit REQUIRED)

add_subdirectory(src)
add_subdirectory(main)
add_subdirectory(tests)
add_subdirectory(lib/Catch2)