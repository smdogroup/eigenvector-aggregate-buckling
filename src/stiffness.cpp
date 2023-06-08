#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace py = pybind11;

template <typename T>
py::array_t<T> _populate_Be(int nelems, T xi, T eta,
                            const py::array_t<double>& xe,
                            const py::array_t<double>& ye, py::array_t<T>& Be) {
  auto xe_data = xe.template unchecked<2>();
  auto ye_data = ye.template unchecked<2>();
  auto Be_data = Be.template mutable_unchecked<3>();

  std::vector<std::vector<std::vector<T>>> J(
      nelems, std::vector<std::vector<T>>(2, std::vector<T>(2, 0.0)));
  std::vector<std::vector<std::vector<T>>> invJ(
      nelems, std::vector<std::vector<T>>(2, std::vector<T>(2, 0.0)));
  std::vector<T> Nxi = {-0.25 * (1.0 - eta), 0.25 * (1.0 - eta),
                        0.25 * (1.0 + eta), -0.25 * (1.0 + eta)};
  std::vector<T> Neta = {-0.25 * (1.0 - xi), -0.25 * (1.0 + xi),
                         0.25 * (1.0 + xi), 0.25 * (1.0 - xi)};

  for (int i = 0; i < nelems; ++i) {
    J[i][0][0] = Nxi[0] * xe_data(i, 0) + Nxi[1] * xe_data(i, 1) +
                 Nxi[2] * xe_data(i, 2) + Nxi[3] * xe_data(i, 3);
    J[i][0][1] = Nxi[0] * ye_data(i, 0) + Nxi[1] * ye_data(i, 1) +
                 Nxi[2] * ye_data(i, 2) + Nxi[3] * ye_data(i, 3);
    J[i][1][0] = Neta[0] * xe_data(i, 0) + Neta[1] * xe_data(i, 1) +
                 Neta[2] * xe_data(i, 2) + Neta[3] * xe_data(i, 3);
    J[i][1][1] = Neta[0] * ye_data(i, 0) + Neta[1] * ye_data(i, 1) +
                 Neta[2] * ye_data(i, 2) + Neta[3] * ye_data(i, 3);
  }

  py::array_t<double> detJ({nelems});
  auto detJ_data = detJ.mutable_unchecked<1>();
  for (int i = 0; i < nelems; i++) {
    detJ_data(i) = J[i][0][0] * J[i][1][1] - J[i][0][1] * J[i][1][0];
    invJ[i][0][0] = J[i][1][1] / detJ_data(i);
    invJ[i][0][1] = -J[i][0][1] / detJ_data(i);
    invJ[i][1][0] = -J[i][1][0] / detJ_data(i);
    invJ[i][1][1] = J[i][0][0] / detJ_data(i);
  }

  std::vector<std::vector<T>> Nx(nelems, std::vector<T>(4, 0.0));
  std::vector<std::vector<T>> Ny(nelems, std::vector<T>(4, 0.0));

  for (int i = 0; i < nelems; i++) {
    for (int j = 0; j < 4; j++) {
      Nx[i][j] = invJ[i][0][0] * Nxi[j] + invJ[i][1][0] * Neta[j];
      Ny[i][j] = invJ[i][0][1] * Nxi[j] + invJ[i][1][1] * Neta[j];
    }
  }

  for (int i = 0; i < nelems; i++) {
    for (int j = 0; j < 4; j++) {
      Be_data(i, 0, j * 2) = Nx[i][j];
      Be_data(i, 1, j * 2 + 1) = Ny[i][j];
      Be_data(i, 2, j * 2) = Ny[i][j];
      Be_data(i, 2, j * 2 + 1) = Nx[i][j];
    }
  }

  return detJ;
}

// Function to assemble the stiffness matrix
py::array_t<double> assemble_stiffness_matrix(
    const py::array_t<double>& X, const py::array_t<int>& conn,
    const py::array_t<double>& rho, const py::array_t<double>& C0,
    double rho0_K, const std::string& ptype_K, double p, double q) {
  auto X_data = X.template unchecked<2>();
  auto conn_data = conn.template unchecked<2>();
  auto rho_data = rho.template unchecked<1>();
  auto C0_data = C0.template unchecked<2>();

  int nelems = conn.shape(0);
  int nnodes = X.shape(0);

  py::array_t<double> rhoE({nelems});
  auto rhoE_data = rhoE.mutable_unchecked<1>();

  for (int i = 0; i < nelems; ++i) {
    rhoE_data(i) =
        0.25 * (rho_data(conn_data(i, 0)) + rho_data(conn_data(i, 1)) +
                rho_data(conn_data(i, 2)) + rho_data(conn_data(i, 3)));
  }

  py::array_t<double> C({nelems, 3, 3});
  auto C_data = C.mutable_unchecked<3>();

  if (ptype_K == "simp") {
    for (int i = 0; i < nelems; ++i) {
      rhoE_data(i) = std::pow(rhoE_data(i), p) + rho0_K;
    }
  } else {  // ramp
    for (int i = 0; i < nelems; ++i) {
      rhoE_data(i) = rhoE_data(i) / (1.0 + q * (1.0 - rhoE_data(i)));
    }
  }

  for (int i = 0; i < nelems; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        C_data(i, j, k) = rhoE_data(i) * C0_data(j, k);
      }
    }
  }

  double gauss_pts[2] = {-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)};

  py::array_t<double> Ke({nelems, 8, 8});
  auto Ke_data = Ke.mutable_unchecked<3>();
  for (int i = 0; i < nelems; ++i) {
    for (int j = 0; j < 8; ++j) {
      for (int k = 0; k < 8; ++k) {
        Ke_data(i, j, k) = 0.0;
      }
    }
  }

  py::array_t<double> Be({nelems, 3, 8});
  auto Be_data = Be.template mutable_unchecked<3>();
  for (int i = 0; i < nelems; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 8; ++k) {
        Be_data(i, j, k) = 0.0;
      }
    }
  }

  py::array_t<double> xe({nelems, 4});
  auto xe_data = xe.mutable_unchecked<2>();

  py::array_t<double> ye({nelems, 4});
  auto ye_data = ye.mutable_unchecked<2>();

  for (int i = 0; i < nelems; i++) {
    for (int j = 0; j < 4; j++) {
      xe_data(i, j) = X_data(conn_data(i, j), 0);
      ye_data(i, j) = X_data(conn_data(i, j), 1);
    }
  }

  for (int jj = 0; jj < 2; jj++) {
    for (int ii = 0; ii < 2; ii++) {
      double xi = gauss_pts[ii];
      double eta = gauss_pts[jj];

      auto detJ = _populate_Be<double>(nelems, xi, eta, xe, ye, Be);

      auto detJ_data = detJ.mutable_unchecked<1>();

      for (int n = 0; n < nelems; n++) {
        for (int j = 0; j < 8; j++) {
          for (int i = 0; i < 3; i++) {
            for (int l = 0; l < 8; l++) {
              for (int k = 0; k < 3; k++) {
                Ke_data(n, j, l) += detJ_data(n) * Be_data(n, i, j) *
                                    C_data(n, i, k) * Be_data(n, k, l);
              }
            }
          }
        }
      }
    }
  }

  return Ke;
}

// define a function x = 2 * x
void multiply_by_two(py::array_t<double>& x) {
  auto x_data = x.mutable_data();

  for (py::ssize_t i = 0; i < x.size(); ++i) {
    x_data[i] *= 2;
  }
}

std::vector<double> multiply_by_two_copy(const std::vector<double>& x) {
  std::vector<double> y(x.size());
  for (int i = 0; i < x.size(); ++i) {
    y[i] = 2 * x[i];
  }
  return y;
}

void multiply_by_two2d(py::array_t<double>& x) {
  auto x_data = x.mutable_data();
  auto shape = x.shape();

  for (py::ssize_t i = 0; i < shape[0]; ++i) {
    for (py::ssize_t j = 0; j < shape[1]; ++j) {
      auto index = i * shape[1] + j;
      x_data[index] *= 2;
    }
  }
}

PYBIND11_MODULE(stiffness, m) {
  m.def("assemble_stiffness_matrix", &assemble_stiffness_matrix,
        "Populate Be using Eigen library");

  m.def("multiply_by_two", &multiply_by_two, "Multiply by two");

  m.def("multiply_by_two_copy", &multiply_by_two_copy, "Multiply by two");

  m.def("multiply_by_two2d", &multiply_by_two2d, "Multiply by two");
}
