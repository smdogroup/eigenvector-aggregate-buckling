#ifndef STRESS_STIFFNESS_H
#define STRESS_STIFFNESS_H

// #include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>
#include <cstring>

#include "converter.h"
#include "utils.hpp"

template <typename T>
View1D<T> populateBeTe(T xi, T eta, const View2D<T>& xe, const View2D<T>& ye, View3D<T>& Be,
                       View4D<T>& Te) {
  const int nelems = xe.extent(0);
  View1D<T> detJ("detJ", nelems);
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

  Kokkos::parallel_for(
      nelems, KOKKOS_LAMBDA(const int n) {
        for (int i = 0; i < 4; ++i) {
          J(n, 0, 0) += Nxi(i) * xe(n, i);
          J(n, 0, 1) += Nxi(i) * ye(n, i);
          J(n, 1, 0) += Neta(i) * xe(n, i);
          J(n, 1, 1) += Neta(i) * ye(n, i);
        }

        detJ(n) = J(n, 0, 0) * J(n, 1, 1) - J(n, 0, 1) * J(n, 1, 0);

        invJ(n, 0, 0) = J(n, 1, 1) / detJ(n);
        invJ(n, 0, 1) = -J(n, 0, 1) / detJ(n);
        invJ(n, 1, 0) = -J(n, 1, 0) / detJ(n);
        invJ(n, 1, 1) = J(n, 0, 0) / detJ(n);

        for (int i = 0; i < 4; ++i) {
          Nx(n, i) = invJ(n, 0, 0) * Nxi(i) + invJ(n, 0, 1) * Neta(i);
          Ny(n, i) = invJ(n, 1, 0) * Nxi(i) + invJ(n, 1, 1) * Neta(i);

          Be(n, 0, i * 2) = Be(n, 2, i * 2 + 1) = Nx(n, i);
          Be(n, 2, i * 2) = Be(n, 1, i * 2 + 1) = Ny(n, i);
        }

        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 4; j++) {
            Te(n, 0, i, j) = Nx(n, i) * Nx(n, j);
            Te(n, 1, i, j) = Ny(n, i) * Ny(n, j);
            Te(n, 2, i, j) = Nx(n, i) * Ny(n, j) + Ny(n, i) * Nx(n, j);
          }
        }
      });

  return detJ;
}

template <typename T, typename D>
View3D<D> computeG(const View1D<D>& rho, const View1D<D>& u, const View3D<T>& detJ,
                   const View5D<T>& Be, const View6D<T>& Te, const View2D<int>& conn,
                   const View2D<T>& C0, const T rho0_K, const char* ptype_K, const double p,
                   const double q) {
  const int nelems = conn.extent(0);
  View3D<D> C("C", nelems, 3, 3);
  View2D<D> ue("ue", nelems, 8);
  View3D<D> Ge("Ge", nelems, 8, 8);

  // Prepare the shape functions
  Kokkos::parallel_for(
      nelems, KOKKOS_LAMBDA(const int n) {
        // Average the density to get the element - wise density
        D rhoE_n = 0.25 * (rho(conn(n, 0)) + rho(conn(n, 1)) + rho(conn(n, 2)) + rho(conn(n, 3)));

        // Compute the constitutivve matrix
        const char* simp = "simp";
        const char* ramp = "ramp";
        for (int i = 0; i < 3; i++) {
          for (int k = 0; k < 3; k++) {
            if (strcmp(ptype_K, simp) == 0) {
              C(n, i, k) = (std::pow(rhoE_n, p) + rho0_K) * C0(i, k);
            } else if (strcmp(ptype_K, ramp) == 0) {
              C(n, i, k) = (rhoE_n / (1.0 + q * (1.0 - rhoE_n))) * C0(i, k);
            } else {
              printf("Penalty type not supported\n");
            }
          }
        }

        // Get the element-wise solution variables
        // Compute the x and y coordinates of each element
        for (int i = 0; i < 4; ++i) {
          ue(n, i * 2) = u(conn(n, i) * 2);
          ue(n, i * 2 + 1) = u(conn(n, i) * 2 + 1);
        }
      });

  for (int jj = 0; jj < 2; jj++) {
    for (int ii = 0; ii < 2; ii++) {
      auto detJ_ij = Kokkos::subview(detJ, ii, jj, Kokkos::ALL());
      auto Be_ij = Kokkos::subview(Be, ii, jj, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
      auto Te_ij =
          Kokkos::subview(Te, ii, jj, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());

      // Compute the stresses in each element
      // s = np.einsum("nij,njk,nk -> ni", C, Be, ue)
      // G0e = np.einsum("n,ni,nijl -> njl", detJ, s, Te)
      Kokkos::parallel_for(
          nelems, KOKKOS_LAMBDA(const int n) {
            for (int i = 0; i < 3; i++) {
              D s_ni = 0.0;
              for (int k = 0; k < 8; k++) {
                for (int j = 0; j < 3; j++) {
                  s_ni += C(n, i, j) * Be_ij(n, j, k) * ue(n, k);
                }
              }

              for (int j = 0; j < 4; j++) {
                for (int l = 0; l < 4; l++) {
                  D temp = detJ_ij(n) * s_ni * Te_ij(n, i, j, l);
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

  returns: dfdu and dfdC
*/
template <typename T>
std::tuple<View1D<T>, View1D<T>, View3D<T>> computeGDerivative(
    const View1D<T>& rho, const View1D<T>& u, const View3D<T>& detJ, const View5D<T>& Be,
    const View6D<T>& Te, const View2D<int>& conn, const View1D<T>& psi, const View1D<T>& phi,
    const View2D<T>& C0, const T rho0_K, const char* ptype_K, const double p, const double q) {
  const int nelems = conn.extent(0);
  const int nnodes = rho.extent(0);

  View1D<T> rhoE("rhoE", nelems);
  View3D<T> C("C", nelems, 3, 3);
  View2D<T> ue("ue", nelems, 8);
  View2D<T> psie("psie", nelems, 8);
  View2D<T> phie("psie", nelems, 8);

  View2D<T> dfdue("dfdue", nelems, 8);
  View1D<T> dfdu("dfdu", 2 * nnodes);
  View3D<T> dfdC("dfdC", nelems, 3, 3);

  // Prepare the shape functions
  Kokkos::parallel_for(
      nelems, KOKKOS_LAMBDA(const int n) {
        // Average the density to get the element - wise density
        rhoE(n) = 0.25 * (rho(conn(n, 0)) + rho(conn(n, 1)) + rho(conn(n, 2)) + rho(conn(n, 3)));

        // Compute the constitutivve matrix
        const char* simp = "simp";
        const char* ramp = "ramp";
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            if (strcmp(ptype_K, simp) == 0) {
              C(n, i, j) = (std::pow(rhoE(n), p) + rho0_K) * C0(i, j);
            } else if (strcmp(ptype_K, ramp) == 0) {
              C(n, i, j) = (rhoE(n) / (1.0 + q * (1.0 - rhoE(n)))) * C0(i, j);
            } else {
              printf("Penalty type not supported\n");
            }
          }
        }

        // Prepare shape functions
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 2; ++j) {
            ue(n, i * 2 + j) = u(conn(n, i) * 2 + j);
            psie(n, i * 2 + j) = psi(conn(n, i) * 2 + j);
            phie(n, i * 2 + j) = phi(conn(n, i) * 2 + j);
          }
        }
      });

  for (int jj = 0; jj < 2; jj++) {
    for (int ii = 0; ii < 2; ii++) {
      auto detJ_ij = Kokkos::subview(detJ, ii, jj, Kokkos::ALL());
      auto Be_ij = Kokkos::subview(Be, ii, jj, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
      auto Te_ij =
          Kokkos::subview(Te, ii, jj, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());

      // dfds = np.einsum( "n,nijl,nj,nl -> ni", detJ, Te, psie[:, ::2],
      // phie[:, ::2]) + np.einsum("n,nijl,nj,nl -> ni", detJ, Te, psie[:,
      // 1::2], phie[:, 1::2] )
      View2D<T> dfds("dfds", nelems, 3);

      Kokkos::parallel_for(
          nelems, KOKKOS_LAMBDA(const int n) {
            for (int i = 0; i < 3; i++) {
              T temp = 0.0;
              for (int j = 0; j < 4; j++) {
                for (int l = 0; l < 4; l++) {
                  temp +=
                      detJ_ij(n) * Te_ij(n, i, j, l) *
                      (psie(n, j * 2) * phie(n, l * 2) + psie(n, j * 2 + 1) * phie(n, l * 2 + 1));
                }
              }
              dfds(n, i) += temp;
            }
          });

      // Add up contributions to d( psi^{T} * G(x, u) * phi ) / du
      // dfdue += np.einsum("nij,nik,nk -> nj", Be, C, dfds)
      Kokkos::parallel_for(
          nelems, KOKKOS_LAMBDA(const int n) {
            for (int j = 0; j < 8; j++) {
              T temp = 0.0;
              for (int i = 0; i < 3; i++) {
                for (int k = 0; k < 3; k++) {
                  temp += Be_ij(n, i, j) * C(n, i, k) * dfds(n, k);
                }
              }
              dfdue(n, j) += temp;
            }
          });

      // Add contributions to the derivative w.r.t. C
      // dfdC += np.einsum("ni,njk,nk -> nij", dfds, Be, ue)
      Kokkos::parallel_for(
          nelems, KOKKOS_LAMBDA(const int n) {
            for (int i = 0; i < 3; i++) {
              for (int j = 0; j < 3; j++) {
                T temp = 0.0;
                for (int k = 0; k < 8; k++) {
                  temp += dfds(n, i) * Be_ij(n, j, k) * ue(n, k);
                }
                dfdC(n, i, j) += temp;
              }
            }
          });
    };
  };

  // // np.add.at(dfdu, 2 * self.conn, dfdue[:, 0::2])
  // // np.add.at(dfdu, 2 * self.conn + 1, dfdue[:, 1::2])
  Kokkos::parallel_for(
      nelems, KOKKOS_LAMBDA(const int n) {
        for (int i = 0; i < 4; i++) {
          Kokkos::atomic_add(&dfdu(conn(n, i) * 2), dfdue(n, i * 2));
          Kokkos::atomic_add(&dfdu(conn(n, i) * 2 + 1), dfdue(n, i * 2 + 1));
        }
      });

  return std::make_tuple(rhoE, dfdu, dfdC);
}

#endif

// template <typename T, typename D>
// View3D<D> computeG(const View2D<T>& X, const View2D<int>& conn, const View1D<D>& rho,
//                    const View1D<D>& u, const View2D<T>& C0, const T rho0_K, const char* ptype_K,
//                    const double p, const double q) {
//   const int nelems = conn.extent(0);
//   View3D<D> C("C", nelems, 3, 3);
//   View2D<T> xe("xe", nelems, 4);
//   View2D<T> ye("ye", nelems, 4);
//   View2D<D> ue("ue", nelems, 8);
//   View3D<T> Be("Be", nelems, 3, 8);
//   View4D<T> Te("Te", nelems, 3, 4, 4);
//   View3D<D> Ge("Ge", nelems, 8, 8);

//   // Compute Gauss quadrature with a 2-point quadrature rule
//   const double gauss_pts[2] = {-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)};

//   // Prepare the shape functions
//   Kokkos::parallel_for(
//       nelems, KOKKOS_LAMBDA(const int n) {
//         // Average the density to get the element - wise density
//         D rhoE_n = 0.25 * (rho(conn(n, 0)) + rho(conn(n, 1)) + rho(conn(n, 2)) + rho(conn(n,
//         3)));

//         // Compute the constitutivve matrix
//         const char* simp = "simp";
//         const char* ramp = "ramp";
//         for (int i = 0; i < 3; i++) {
//           for (int k = 0; k < 3; k++) {
//             if (strcmp(ptype_K, simp) == 0) {
//               C(n, i, k) = (std::pow(rhoE_n, p) + rho0_K) * C0(i, k);
//             } else if (strcmp(ptype_K, ramp) == 0) {
//               C(n, i, k) = (rhoE_n / (1.0 + q * (1.0 - rhoE_n))) * C0(i, k);
//             } else {
//               printf("Penalty type not supported\n");
//             }
//           }
//         }

//         // Get the element-wise solution variables
//         // Compute the x and y coordinates of each element
//         for (int i = 0; i < 4; ++i) {
//           xe(n, i) = X(conn(n, i), 0);
//           ye(n, i) = X(conn(n, i), 1);
//           ue(n, i * 2) = u(conn(n, i) * 2);
//           ue(n, i * 2 + 1) = u(conn(n, i) * 2 + 1);
//         }
//       });

//   for (int jj = 0; jj < 2; jj++) {
//     for (int ii = 0; ii < 2; ii++) {
//       T xi = gauss_pts[ii];
//       T eta = gauss_pts[jj];

//       auto detJ = populateBeTe(xi, eta, xe, ye, Be, Te);

//       // Compute the stresses in each element
//       // s = np.einsum("nij,njk,nk -> ni", C, Be, ue)
//       // G0e = np.einsum("n,ni,nijl -> njl", detJ, s, Te)
//       Kokkos::parallel_for(
//           nelems, KOKKOS_LAMBDA(const int n) {
//             for (int i = 0; i < 3; i++) {
//               D s_ni = 0.0;
//               for (int k = 0; k < 8; k++) {
//                 for (int j = 0; j < 3; j++) {
//                   s_ni += C(n, i, j) * Be(n, j, k) * ue(n, k);
//                 }
//               }

//               for (int j = 0; j < 4; j++) {
//                 for (int l = 0; l < 4; l++) {
//                   D temp = detJ(n) * s_ni * Te(n, i, j, l);
//                   Ge(n, j * 2, l * 2) += temp;
//                   Ge(n, j * 2 + 1, l * 2 + 1) += temp;
//                 }
//               }
//             }
//           });
//     };
//   };

//   return Ge;
// }

// template <typename T>
// std::tuple<View1D<T>, View1D<T>, View3D<T>> computeGDerivative(
//     const View2D<T>& X, const View2D<int>& conn, const View1D<T>& rho, const View1D<T>& u,
//     const View1D<T>& psi, const View1D<T>& phi, const View2D<T>& C0, const T rho0_K,
//     const char* ptype_K, const double p, const double q) {
//   const int nelems = conn.extent(0);
//   const int nnodes = X.extent(0);

//   View1D<T> rhoE("rhoE", nelems);
//   View3D<T> C("C", nelems, 3, 3);
//   View2D<T> xe("xe", nelems, 4);
//   View2D<T> ye("ye", nelems, 4);
//   View2D<T> ue("ue", nelems, 8);
//   View2D<T> psie("psie", nelems, 8);
//   View2D<T> phie("psie", nelems, 8);
//   View3D<T> Be("Be", nelems, 3, 8);
//   View4D<T> Te("Te", nelems, 3, 4, 4);

//   View2D<T> dfdue("dfdue", nelems, 8);
//   View1D<T> dfdu("dfdu", 2 * nnodes);
//   View3D<T> dfdC("dfdC", nelems, 3, 3);

//   // Compute Gauss quadrature with a 2-point quadrature rule
//   const double gauss_pts[2] = {-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)};

//   // Prepare the shape functions
//   Kokkos::parallel_for(
//       nelems, KOKKOS_LAMBDA(const int n) {
//         // Average the density to get the element - wise density
//         rhoE(n) = 0.25 * (rho(conn(n, 0)) + rho(conn(n, 1)) + rho(conn(n, 2)) + rho(conn(n, 3)));

//         // Compute the constitutivve matrix
//         const char* simp = "simp";
//         const char* ramp = "ramp";
//         for (int i = 0; i < 3; i++) {
//           for (int j = 0; j < 3; j++) {
//             if (strcmp(ptype_K, simp) == 0) {
//               C(n, i, j) = (std::pow(rhoE(n), p) + rho0_K) * C0(i, j);
//             } else if (strcmp(ptype_K, ramp) == 0) {
//               C(n, i, j) = (rhoE(n) / (1.0 + q * (1.0 - rhoE(n)))) * C0(i, j);
//             } else {
//               printf("Penalty type not supported\n");
//             }
//           }
//         }

//         for (int i = 0; i < 4; ++i) {
//           xe(n, i) = X(conn(n, i), 0);
//           ye(n, i) = X(conn(n, i), 1);
//           ue(n, i * 2) = u(conn(n, i) * 2);
//           ue(n, i * 2 + 1) = u(conn(n, i) * 2 + 1);
//           psie(n, i * 2) = psi(conn(n, i) * 2);
//           psie(n, i * 2 + 1) = psi(conn(n, i) * 2 + 1);
//           phie(n, i * 2) = phi(conn(n, i) * 2);
//           phie(n, i * 2 + 1) = phi(conn(n, i) * 2 + 1);
//         }
//       });

//   for (int jj = 0; jj < 2; jj++) {
//     for (int ii = 0; ii < 2; ii++) {
//       T xi = gauss_pts[ii];
//       T eta = gauss_pts[jj];

//       auto detJ = populateBeTe(xi, eta, xe, ye, Be, Te);

//       // dfds = np.einsum( "n,nijl,nj,nl -> ni", detJ, Te, psie[:, ::2],
//       // phie[:, ::2]) + np.einsum("n,nijl,nj,nl -> ni", detJ, Te, psie[:,
//       // 1::2], phie[:, 1::2] )
//       View2D<T> dfds("dfds", nelems, 3);

//       Kokkos::parallel_for(
//           Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nelems, 3}),
//           KOKKOS_LAMBDA(const int n, const int i) {
//             for (int j = 0; j < 4; j++) {
//               for (int l = 0; l < 4; l++) {
//                 dfds(n, i) +=
//                     detJ(n) * Te(n, i, j, l) *
//                     (psie(n, j * 2) * phie(n, l * 2) + psie(n, j * 2 + 1) * phie(n, l * 2 + 1));
//               }
//             }
//             // }
//           });

//       // Add up contributions to d( psi^{T} * G(x, u) * phi ) / du
//       // dfdue += np.einsum("nij,nik,nk -> nj", Be, C, dfds)
//       Kokkos::parallel_for(
//           Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nelems, 8}), KOKKOS_LAMBDA(int n, int
//           j) {
//             for (int i = 0; i < 3; i++) {
//               for (int k = 0; k < 3; k++) {
//                 dfdue(n, j) += Be(n, i, j) * C(n, i, k) * dfds(n, k);
//               }
//             }
//           });

//       // Add contributions to the derivative w.r.t. C
//       // dfdC += np.einsum("ni,njk,nk -> nij", dfds, Be, ue)
//       Kokkos::parallel_for(
//           Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nelems, 3, 3}),
//           KOKKOS_LAMBDA(int n, int i, int j) {
//             for (int k = 0; k < 8; k++) {
//               dfdC(n, i, j) += dfds(n, i) * Be(n, j, k) * ue(n, k);
//             }
//           });
//     };
//   };

//   // // np.add.at(dfdu, 2 * self.conn, dfdue[:, 0::2])
//   // // np.add.at(dfdu, 2 * self.conn + 1, dfdue[:, 1::2])
//   Kokkos::parallel_for(
//       nelems, KOKKOS_LAMBDA(const int n) {
//         for (int i = 0; i < 4; i++) {
//           Kokkos::atomic_add(&dfdu(conn(n, i) * 2), dfdue(n, i * 2));
//           Kokkos::atomic_add(&dfdu(conn(n, i) * 2 + 1), dfdue(n, i * 2 + 1));
//         }
//       });

//   return std::make_tuple(rhoE, dfdu, dfdC);
// }