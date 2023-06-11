cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(buckling, LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set warning flags
add_compile_options(
  -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-sign-compare
  )

# Find the required packages
find_package(pybind11 REQUIRED)
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)

# Set the source files and add module
set(SOURCE_FILES src/warpper.cpp)
pybind11_add_module(kokkos MODULE ${SOURCE_FILES})

# Set the output directory for the shared library
set_target_properties(kokkos PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Link the required libraries
target_link_libraries(kokkos PRIVATE Kokkos::kokkos Kokkos::kokkoskernels)

# Add optimization flags specific to the target architecture
target_compile_options(kokkos PRIVATE -march=native -mtune=native -O3)