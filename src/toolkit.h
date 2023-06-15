#ifndef TOOLKIT_H
#define TOOLKIT_H

// Macro to print the line number with pass status
#define checks() printLineStatus(__LINE__, true)

#define tick(msg) startTimer(msg);
#define tock(...) reportTimer(__VA_ARGS__);

#include <Kokkos_Core.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::time_point<Clock> TimePoint;
typedef std::chrono::duration<double> Duration;

TimePoint startTime;
TimePoint endTime;

void startTimer(const char* msg) { startTime = Clock::now(); }

double getElapsedTime() {
  Duration d = Clock::now() - startTime;
  return d.count();
}

void reportTimer(const char* msg = "") {
  double elapsed = getElapsedTime();
  std::cout << std::setprecision(5) << std::fixed;
  if (elapsed < 1e-6)
    std::cout << msg << ": "
              << "\033[32m" << elapsed * 1e6 << " us\033[0m" << std::endl;
  else if (elapsed < 1e-3)
    std::cout << msg << ": "
              << "\033[32m" << elapsed * 1e3 << " ms\033[0m" << std::endl;
  else
    std::cout << msg << ": "
              << "\033[32m" << elapsed << " s\033[0m" << std::endl;
}

// Function to print the line number and status
void printLineStatus(int lineNumber, bool passed) {
  std::cout << "Line: "
            << "\033[32m" << lineNumber << " ";
  if (passed) {
    std::cout << "pass";
  } else {
    std::cout << "not pass";
  }
  std::cout << "\033[0m" << std::endl;
}

#endif  // TOOLKIT_H