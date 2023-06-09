#pragma once

#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_mult.hpp>
#include <KokkosBlas1_reciprocal.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas1_update.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <Kokkos_Core.hpp>

#include "wrapper.h"

template <typename T>
auto populateBe(int nelems, T xi, T eta, const View2D<T>& xe,
                const View2D<T>& ye, View3D<T>& Be) {
  View1D<T> detJ("detJ", nelems);
  View1D<T> invdetJ("invdetJ", nelems);
  View3D<T> J("J", nelems, 2, 2);
  View3D<T> invJ("invJ", nelems, 2, 2);

  View1D<T> Nxi("Nxi", 4);
  View1D<T> Neta("Neta", 4);

  View2D<T> Nx("Nx", nelems, 4);
  View2D<T> Ny("Ny", nelems, 4);

  Nxi(0) = -0.25 * (1.0 - eta);
  Nxi(1) = 0.25 * (1.0 - eta);
  Nxi(2) = 0.25 * (1.0 + eta);
  Nxi(3) = -0.25 * (1.0 + eta);

  Neta(0) = -0.25 * (1.0 - xi);
  Neta(1) = -0.25 * (1.0 + xi);
  Neta(2) = 0.25 * (1.0 + xi);
  Neta(3) = 0.25 * (1.0 - xi);

  auto J00 = Kokkos::subview(J, Kokkos::ALL(), 0, 0);
  auto J01 = Kokkos::subview(J, Kokkos::ALL(), 0, 1);
  auto J10 = Kokkos::subview(J, Kokkos::ALL(), 1, 0);
  auto J11 = Kokkos::subview(J, Kokkos::ALL(), 1, 1);

  // KokkosBlas::gemv (mode, alpha, A, x, beta, y)
  // y[i] = beta * y[i] + alpha * SUM_j(A[i,j] * x[j])
  KokkosBlas::gemv("N", 1.0, xe, Nxi, 0.0, J00);
  KokkosBlas::gemv("N", 1.0, xe, Neta, 0.0, J01);
  KokkosBlas::gemv("N", 1.0, ye, Nxi, 0.0, J10);
  KokkosBlas::gemv("N", 1.0, ye, Neta, 0.0, J11);

  // detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
  // KokkosBlas::mult(gamma,y,alpha,a,x):
  // y[i] <- gamma*y[i] + alpha*a[i]*x[i]
  KokkosBlas::mult(0.0, detJ, 1.0, J00, J11);
  KokkosBlas::mult(1.0, detJ, -1.0, J01, J10);

  auto invJ00 = Kokkos::subview(invJ, Kokkos::ALL(), 0, 0);
  auto invJ01 = Kokkos::subview(invJ, Kokkos::ALL(), 0, 1);
  auto invJ10 = Kokkos::subview(invJ, Kokkos::ALL(), 1, 0);
  auto invJ11 = Kokkos::subview(invJ, Kokkos::ALL(), 1, 1);

  KokkosBlas::reciprocal(invdetJ, detJ);
  KokkosBlas::mult(0.0, invJ00, 1.0, J11, invdetJ);
  KokkosBlas::mult(0.0, invJ01, -1.0, J01, invdetJ);
  KokkosBlas::mult(0.0, invJ10, -1.0, J10, invdetJ);
  KokkosBlas::mult(0.0, invJ11, 1.0, J00, invdetJ);

  // KokkosBlas::update(alpha,x,beta,y,gamma,z)
  // z[i] <- gamma*z[i] + alpha*x[i] + beta*y[i]
  // Nx(i, 0) = invJ(i, 0, 0) * Nxi[0] + invJ(i, 1, 0) * Neta[0]
  // Nx = np.outer(invJ[:, 0, 0], Nxi) + np.outer(invJ[:, 1, 0], Neta)
  // Ny = np.outer(invJ[:, 0, 1], Nxi) + np.outer(invJ[:, 1, 1], Neta)
  for (int i = 0; i < 4; ++i) {
    KokkosBlas::update(Nxi(i), invJ00, Neta(i), invJ10, 0.0,
                       Kokkos::subview(Nx, Kokkos::ALL(), i));
    KokkosBlas::update(Nxi(i), invJ01, Neta(i), invJ11, 0.0,
                       Kokkos::subview(Ny, Kokkos::ALL(), i));
  }

  Kokkos::parallel_for(
      "PopulateBe", nelems, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < 4; j++) {
          Be(i, 0, j * 2) = Nx(i, j);
          Be(i, 1, j * 2 + 1) = Ny(i, j);
          Be(i, 2, j * 2) = Ny(i, j);
          Be(i, 2, j * 2 + 1) = Nx(i, j);
        }
      });

  return detJ;
}

template <typename T>
View3D<T> computeElementStiffnesses(const View2D<T>& X, const View2D<int>& conn,
                                    const View1D<T>& rho, const View2D<T>& C0,
                                    const T rho0_K, const std::string& ptype_K,
                                    const double p, const double q) {
  const int nelems = conn.extent(0);
  const int num_nodes = conn.extent(1);

  View1D<T> rhoE("rhoE", nelems);
  View3D<T> C("C", nelems, 3, 3);
  View3D<T> Ke("Ke", nelems, 8, 8);

  Kokkos::parallel_for(
      "AverageDensity", RangePolicy(0, nelems), KOKKOS_LAMBDA(const int i) {
        rhoE(i) = 0.25 * (rho(conn(i, 0)) + rho(conn(i, 1)) + rho(conn(i, 2)) +
                          rho(conn(i, 3)));
      });

  // Compute the element stiffnesses
  if (ptype_K == "simp") {
    Kokkos::parallel_for(
        nelems,
        KOKKOS_LAMBDA(int i) { rhoE(i) = std::pow(rhoE(i), p) + rho0_K; });
  } else {  // ramp
    Kokkos::parallel_for(
        nelems, KOKKOS_LAMBDA(int i) {
          rhoE(i) = rhoE(i) / (1.0 + q * (1.0 - rhoE(i)));
        });
  }

  // C = outer(rhoE, C0)
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 3; ++k) {
      auto C_jk = Kokkos::subview(C, Kokkos::ALL(), j, k);
      KokkosBlas::scal(C_jk, C0(j, k), rhoE);
    }
  }

  // Compute the element stiffness matrix
  const T gauss_pts[2] = {-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)};

  View2D<T> xe("xe", nelems, num_nodes);
  View2D<T> ye("ye", nelems, num_nodes);

  Kokkos::parallel_for(
      "ComputeElementCoordinates", RangePolicy(0, nelems),
      KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < num_nodes; ++j) {
          xe(i, j) = X(conn(i, j), 0);
          ye(i, j) = X(conn(i, j), 1);
        }
      });

  for (int jj = 0; jj < 2; jj++) {
    for (int ii = 0; ii < 2; ii++) {
      T xi = gauss_pts[ii];
      T eta = gauss_pts[jj];

      View3D<T> Be("Be", nelems, 3, 8);
      auto detJ = populateBe(nelems, xi, eta, xe, ye, Be);

      Kokkos::parallel_for(
          "ComputeKe", RangePolicy(0, nelems), KOKKOS_LAMBDA(const int n) {
            for (int j = 0; j < 8; j++) {
              for (int i = 0; i < 3; i++) {
                for (int l = 0; l < 8; l++) {
                  for (int k = 0; k < 3; k++) {
                    Ke(n, j, l) +=
                        detJ(n) * Be(n, i, j) * C(n, i, k) * Be(n, k, l);
                  }
                }
              }
            }
          });
    };
  };

  return Ke;
}

// convert the data from pyarray to kokkos view by calling the function
// convertPyArrayToView and then call the function computeElementStiffnesses
// to compute the element stiffnesses
template <typename T>
py::array_t<T> assembleStiffnessMatrix(
    py::array_t<T> X_py, py::array_t<int> conn_py, py::array_t<T> rho_py,
    py::array_t<T> C0_py, T rho0_K, std::string ptype_K, double p, double q) {
  auto X = numpyArrayToView2D<T>(X_py);
  auto conn = numpyArrayToView2D<int>(conn_py);
  auto rho = numpyArrayToView1D<T>(rho_py);
  auto C0 = numpyArrayToView2D<T>(C0_py);

  auto Ke =
      computeElementStiffnesses<T>(X, conn, rho, C0, rho0_K, ptype_K, p, q);

  return viewToNumpyArray3D<T>(Ke);
}