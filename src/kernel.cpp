#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;
namespace py = pybind11;

typedef Eigen::SparseMatrix<double> SpMat;

