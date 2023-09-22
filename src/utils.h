#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP

// Macro to print the line number with pass status
#define check() lineStatus(__LINE__, true)
#define tick(msg) startTimer(msg);
#define tock(...) reportTimer(__VA_ARGS__);

// #include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
// #include <iomanip>
// #include <iostream>
// #include <iterator>
#include <map>
#include <random>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::time_point<Clock> TimePoint;
typedef std::chrono::duration<double> Duration;

TimePoint startTime;
TimePoint endTime;
std::map<std::string, double> cumulativeTimes;  // Map to store cumulative times

void startTimer(const char* msg) {
#ifdef KOKKOS_ENABLE_CUDA
  Kokkos::fence();
#endif
  startTime = Clock::now();
}

double getElapsedTime() {
  Duration d = Clock::now() - startTime;
  return d.count();
}

void reportTimer(const char* msg = "") {
#ifdef KOKKOS_ENABLE_CUDA
  Kokkos::fence();
#endif
  double elapsed = getElapsedTime();
  // Add the current elapsed time to the cumulative time for this message
  cumulativeTimes[std::string(msg)] += elapsed;
  printf("%s: ", msg);
  printf("\033[32m%.8f s\033[0m\n", cumulativeTimes[std::string(msg)]);
}

// Function to print the line number and status
void lineStatus(int lineNumber, bool passed) {
  printf("Line: \033[32m%d ", lineNumber);
  if (passed) {
    printf("pass");
  } else {
    printf("not pass");
  }
  printf("\033[0m\n");
}

/**
 * Fill a nxn matrix with random values between min and max
 *
 * Input:
 *   T A: nxn matrix       - row major
 *   T min: minimum value  - default 0.0
 *   T max: maximum value  - default 1.0
 */
template <typename T>
void randFill(T* A, int N, T min = -1.0, T max = 1.0) {
  static std::random_device rd;   // only need to initialize it once
  static std::mt19937 mte(rd());  // this is a relative big object to create

  std::uniform_real_distribution<T> dist(min, max);

  for (int i = 0; i < N; ++i) {
    std::generate(A + i * N, A + (i + 1) * N, [&]() { return dist(mte); });
  }
}

/*  Usage:
    // Assuming A is a 5x5 matrix
    int N = 5;
    int M = 5;
    double A[N * M] = {
        // ... (matrix elements)
    };

    // Print a single element at [2, 2]
    printmat("Single Element", A, 5, 5, std::make_pair(2, 2));

    // Print the first row only (single row interval [0, 1))
    printmat("First Row", A, 5, 5, std::make_pair(0, 1), std::make_pair(0, 5));

    // Print the third column only (single column interval [0, 1))
    printmat("Third Column", A, 5, 5, std::make_pair(0, 5), std::make_pair(2,
   3));

*/
template <typename T>
void printmat(const char* name, const T* A, int totalRows, int totalCols,
              const std::pair<int, int>& N_interval,
              const std::pair<int, int>& M_interval = std::make_pair(0, 1), int layout = 0) {
  int N0 = N_interval.first;
  int N1 = N_interval.second;
  int M0 = M_interval.first;
  int M1 = M_interval.second;

  // Reset N1 and M1 if they are larger than the total number of rows and
  // columns
  if (N1 + 1 > totalRows) {
    printf(
        " \033[31mWarning: N1 is larger than the total number of rows. "
        "Resetting N1 to %d\033[0m\n",
        totalRows - 1);
    N1 = totalRows - 1;
  }
  if (M1 + 1 > totalCols) {
    printf(
        " \033[31mWarning: M1 is larger than the total number of columns. "
        "Resetting M1 to %d\033[0m\n",
        totalCols - 1);
    M1 = totalCols - 1;
  }

  if (N0 > N1) {
    printf(
        " \033[31mWarning: N0 is larger than N1. "
        "Resetting N0 to %d\033[0m\n",
        N1);
    N0 = N1;
  }
  if (M0 > M1) {
    printf(
        " \033[31mWarning: M0 is larger than M1. "
        "Resetting M0 to %d\033[0m\n",
        M1);
    M0 = M1;
  }

  int actual_N = totalRows > 0 ? totalRows : N1 - N0;
  int actual_M = totalCols > 0 ? totalCols : M1 - M0;

  printf("Matrix: \033[32m%s\033[0m\n", name);
  for (int i = N0; i <= N1; ++i) {
    printf("  |");
    for (int j = M0; j <= M1; ++j) {
      if (layout == 0)
        printf("%12.8f ", A[i * actual_M + j]);
      else if (layout == 1)
        printf("%12.8f ", A[i + j * actual_N]);
    }
    printf("|\n");
  }
  printf("\n");
}

template <typename T>
void printmat(const char* name, T* A, int N = 5, int M = 5, int layout = 0) {
  printf("Matrix: \033[32m%s\033[0m\n", name);
  for (int i = 0; i < N; ++i) {
    printf("  |");
    for (int j = 0; j < M; ++j) {
      if (layout == 0)
        printf("%12.8f ", A[i * M + j]);
      else if (layout == 1)
        printf("%12.8f ", A[i + j * N]);
    }
    printf("|\n");
  }
  printf("\n");
}

template <typename container>
void printmat(const char* name, container& A, int N = 5, int M = 5) {
  printf("Matrix: \033[32m%s\033[0m\n", name);
  for (int i = 0; i < N; ++i) {
    printf("  |");
    for (int j = 0; j < M; ++j) {
      printf("%8.5f ", A(i, j));
    }
    printf("|\n");
  }
}

// template <typename Container>
// void printmat(const char* name, Container& A, int N=5) {
//   printf("Matrix: \033[32m%s\033[0m\n", name);
//   for (auto it_row = A.begin(); it_row != A.end(); ++it_row) {
//     printf("  |");
//     for (auto it_col = it_row->begin(); it_col != it_row->end(); ++it_col)
//     {
//         printf("%9.5f ", *it_col);
//     }
//     printf("|\n");
//   }
//   printf("\n");
// }

#endif  // UTILS_HPP