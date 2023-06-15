#ifndef STRESS_STIFFNESS_H
#define STRESS_STIFFNESS_H

#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_mult.hpp>
#include <KokkosBlas1_reciprocal.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas1_update.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <Kokkos_Core.hpp>

#include "converter.h"
#include "toolkit.h"

template <typename T>
auto populateBeTe(T xi, T eta, const View2D<T>& xe, const View2D<T>& ye,
                  View3D<T>& Be, View4D<T>& Te) {
  const int nelems = xe.extent(0);
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
      "PopulateBe", nelems, KOKKOS_LAMBDA(const int n) {
        for (int i = 0; i < 4; i++) {
          Be(n, 0, i * 2) = Nx(n, i);
          Be(n, 1, i * 2 + 1) = Ny(n, i);
          Be(n, 2, i * 2) = Ny(n, i);
          Be(n, 2, i * 2 + 1) = Nx(n, i);
        }
      });

  // for n in range(nelems):
  // Te [n, 0, :, :] = outer(Nx [n, :], Nx [n, :])
  // Te [n, 1, :, :] = outer(Ny [n, :], Ny [n, :])
  // Te [n, 2, :, :] = outer(Nx [n, :], Ny [n, :]) + outer(Ny [n, :], Nx [n, :])
  Kokkos::parallel_for(
      "PopulateTe", nelems, KOKKOS_LAMBDA(const int n) {
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 4; j++) {
            Te(n, 0, i, j) = Nx(n, i) * Nx(n, j);
            Te(n, 1, i, j) = Ny(n, i) * Ny(n, j);
            Te(n, 2, i, j) = Nx(n, i) * Ny(n, j) + Ny(n, i) * Nx(n, j);
          }
        }
      });

  return detJ;
}

template <typename T>
View3D<T> computeG(const View2D<T>& X, const View2D<int>& conn,
                   const View1D<T>& rho, const View1D<T>& u,
                   const View2D<T>& C0, const T rho0_K,
                   const std::string& ptype_K, const double p, const double q) {
  const int nelems = conn.extent(0);
  View3D<T> C("C", nelems, 3, 3);
  View2D<T> xe("xe", nelems, 4);
  View2D<T> ye("ye", nelems, 4);
  View2D<T> ue("ue", nelems, 8);
  View3D<T> Be("Be", nelems, 3, 8);
  View4D<T> Te("Te", nelems, 3, 4, 4);
  View3D<T> Ge("Ge", nelems, 8, 8);

  // Compute Gauss quadrature with a 2-point quadrature rule
  const double gauss_pts[2] = {-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)};

  // Prepare the shape functions
  Kokkos::parallel_for(
      nelems, KOKKOS_LAMBDA(const int n) {
        // Average the density to get the element - wise density
        T rhoE_n = 0.25 * (rho(conn(n, 0)) + rho(conn(n, 1)) + rho(conn(n, 2)) +
                           rho(conn(n, 3)));

        // Compute the constitutivve matrix
        std::string simp = "simp";
        std::string ramp = "ramp";
        for (int i = 0; i < 3; i++) {
          for (int k = 0; k < 3; k++) {
            if (ptype_K == simp) {
              C(n, i, k) = (std::pow(rhoE_n, p) + rho0_K) * C0(i, k);
            } else if (ptype_K == ramp) {
              C(n, i, k) = (rhoE_n / (1.0 + q * (1.0 - rhoE_n))) * C0(i, k);
            } else {
              printf("Penalty type not supported\n");
              exit(1);
            }
          }
        }

        // Get the element-wise solution variables
        // Compute the x and y coordinates of each element
        for (int i = 0; i < 4; ++i) {
          xe(n, i) = X(conn(n, i), 0);
          ye(n, i) = X(conn(n, i), 1);
          ue(n, i * 2) = u(conn(n, i) * 2);
          ue(n, i * 2 + 1) = u(conn(n, i) * 2 + 1);
        }
      });

  for (int jj = 0; jj < 2; jj++) {
    for (int ii = 0; ii < 2; ii++) {
      T xi = gauss_pts[ii];
      T eta = gauss_pts[jj];

      auto detJ = populateBeTe(xi, eta, xe, ye, Be, Te);

      // Compute the stresses in each element
      // s = np.einsum("nij,njk,nk -> ni", C, Be, ue)
      // G0e = np.einsum("n,ni,nijl -> njl", detJ, s, Te)
      Kokkos::parallel_for(
          nelems, KOKKOS_LAMBDA(const int n) {
            for (int i = 0; i < 3; i++) {
              T s_ni = 0.0;
              for (int k = 0; k < 8; k++) {
                for (int j = 0; j < 3; j++) {
                  s_ni += C(n, i, j) * Be(n, j, k) * ue(n, k);
                }
              }

              for (int j = 0; j < 4; j++) {
                for (int l = 0; l < 4; l++) {
                  T temp = detJ(n) * s_ni * Te(n, i, j, l);
                  Ge(n, j * 2, l * 2) += temp;
                  Ge(n, j * 2 + 1, l * 2 + 1) += temp;
                }
              }
            }
          });
    };
  };

  return Ge;
}

/*
  Compute the derivative of psi^{T} * G(u, x) * phi using the adjoint method.

  Note "solver" returns the solution of the system of equations

  K * sol = rhs

  Given the right-hand-side rhs. ie. sol = solver(rhs)
*/
template <typename T>
View1D<T> computeGDerivative(const View2D<T>& X, const View2D<int>& conn,
                             const View1D<T>& rho, const View1D<T>& u,
                             const View1D<T>& psi, const View1D<T>& phi,
                             const View2D<T>& C0, const View1D<T>& reduced,
                             const T rho0_K, const std::string& ptype_K,
                             const double p, const double q) {
  const int nelems = conn.extent(0);
  const int nnodes = X.extent(0);
  const int nreduced = reduced.extent(0);

  View1D<T> rhoE("rhoE", nelems);
  View3D<T> C("C", nelems, 3, 3);
  View2D<T> xe("xe", nelems, 4);
  View2D<T> ye("ye", nelems, 4);
  View2D<T> ue("ue", nelems, 8);
  View2D<T> psie("psie", nelems, 8);
  View2D<T> phie("psie", nelems, 8);
  View3D<T> Be("Be", nelems, 3, 8);
  View4D<T> Te("Te", nelems, 3, 4, 4);

  View2D<T> dfdue("dfdue", nelems, 8);
  View1D<T> dfdu("dfdu", 2 * nnodes);
  View1D<T> dfdur("dfdur", nreduced);
  View3D<T> dfdC("dfdC", nelems, 3, 3);
  View1D<T> dG("dG", nnodes);

  // Compute Gauss quadrature with a 2-point quadrature rule
  const double gauss_pts[2] = {-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)};

  // Prepare the shape functions
  Kokkos::parallel_for(
      nelems, KOKKOS_LAMBDA(const int n) {
        // Average the density to get the element - wise density
        rhoE(n) = 0.25 * (rho(conn(n, 0)) + rho(conn(n, 1)) + rho(conn(n, 2)) +
                          rho(conn(n, 3)));

        // Compute the constitutivve matrix
        std::string simp = "simp";
        std::string ramp = "ramp";
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            if (ptype_K == simp) {
              C(n, i, j) = (std::pow(rhoE(n), p) + rho0_K) * C0(i, j);
            } else if (ptype_K == ramp) {
              C(n, i, j) = (rhoE(n) / (1.0 + q * (1.0 - rhoE(n)))) * C0(i, j);
            } else {
              printf("Penalty type not supported\n");
              exit(1);
            }
          }
        }

        for (int i = 0; i < 4; ++i) {
          xe(n, i) = X(conn(n, i), 0);
          ye(n, i) = X(conn(n, i), 1);
          ue(n, i * 2) = u(conn(n, i) * 2);
          ue(n, i * 2 + 1) = u(conn(n, i) * 2 + 1);
          psie(n, i * 2) = psi(conn(n, i) * 2);
          psie(n, i * 2 + 1) = psi(conn(n, i) * 2 + 1);
          phie(n, i * 2) = phi(conn(n, i) * 2);
          phie(n, i * 2 + 1) = phi(conn(n, i) * 2 + 1);
        }
      });

  for (int jj = 0; jj < 2; jj++) {
    for (int ii = 0; ii < 2; ii++) {
      T xi = gauss_pts[ii];
      T eta = gauss_pts[jj];

      auto detJ = populateBeTe(xi, eta, xe, ye, Be, Te);

      Kokkos::parallel_for(
          nelems, KOKKOS_LAMBDA(const int n) {
            // dfds = np.einsum( "n,nijl,nj,nl -> ni", detJ, Te, psie[:, ::2],
            // phie[:, ::2]) + np.einsum("n,nijl,nj,nl -> ni", detJ, Te, psie[:,
            // 1::2], phie[:, 1::2] )
            T dfds_ni = 0.0;
            for (int i = 0; i < 3; i++) {
              for (int j = 0; j < 4; j++) {
                for (int l = 0; l < 4; l++) {
                  dfds_ni = detJ(n) * Te(n, i, j, l) *
                            (psie(n, j * 2) * phie(n, l * 2) +
                             psie(n, j * 2 + 1) * phie(n, l * 2 + 1));
                }
              }
            }

            // Add up contributions to d( psi^{T} * G(x, u) * phi ) / du
            // dfdue += np.einsum("nij,nik,nk -> nj", Be, C, dfds)
            for (int i = 0; i < 3; i++) {
              for (int j = 0; j < 8; j++) {
                for (int k = 0; k < 8; k++) {
                  dfdue(n, j) += Be(n, i, j) * C(n, i, k) * dfds_ni;
                }
              }
            }

            // Add contributions to the derivative w.r.t. C
            // dfdC += np.einsum("ni,njk,nk -> nij", dfds, Be, ue)
            for (int i = 0; i < 3; i++) {
              for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 8; k++) {
                  dfdC(n, i, j) += dfds_ni * Be(n, i, k) * ue(n, k);
                }
              }
            }
          });
    };
  };

  // // np.add.at(dfdu, 2 * self.conn, dfdue[:, 0::2])
  // // np.add.at(dfdu, 2 * self.conn + 1, dfdue[:, 1::2])
  // Kokkos::parallel_for(
  //     nelems, KOKKOS_LAMBDA(const int n) {
  //       for (int i = 0; i < 4; i++) {
  //         dfdu(conn(n, i) * 2) += dfdue(n, i * 2);
  //         dfdu(conn(n, i) * 2 + 1) += dfdue(n, i * 2 + 1);
  //       }
  //     });

  // // dfdur = dfdu[reduced] where dfdur has same size of reduced
  // Kokkos::parallel_for(
  //     nreduced, KOKKOS_LAMBDA(const int n) { dfdur(n) = dfdu(reduced(n)); });

  // Kokkos::parallel_for(
  //     nelems, KOKKOS_LAMBDA(const int n) {
  //       T drhoE_n = 0.0;
  //       for (int i = 0; i < 3; i++) {
  //         for (int j = 0; j < 3; j++) {
  //           drhoE_n += dfdC(n, i, j) * C0(i, j);
  //         }
  //       }

  //       // Average the density to get the element - wise density
  //       T rhoE_n = 0.25 * (rho(conn(n, 0)) + rho(conn(n, 1)) + rho(conn(n,
  //       2)) +
  //                          rho(conn(n, 3)));

  //       // Penalize the stiffness matrix
  //       if (ptype_K == "simp") {
  //         drhoE_n *= p * pow(rhoE_n, p - 1.0);
  //       } else if (ptype_K == "ramp") {
  //         drhoE_n *= (1.0 + q) / pow((1.0 + q * (1.0 - rhoE_n)), 2.0);
  //       } else {
  //         std::cout << "Penalty type not supported" << std::endl;
  //       }

  //       for (int i = 0; i < 4; i++) {
  //         Kokkos::atomic_add(&dG(conn(n, i)), 0.25 * drhoE_n);
  //       }
  //     });

  return dG;
}

#endif