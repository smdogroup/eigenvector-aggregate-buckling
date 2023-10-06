#ifndef STIFFNESS_H
#define STIFFNESS_H

// #include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>
#include <cstring>

#include "converter.h"
#include "utils.hpp"

template <typename T>
View1D<T> populateBe(T xi, T eta, const View2D<T>& xe, const View2D<T>& ye, View3D<T>& Be) {
  const int nelems = xe.extent(0);
  View1D<T> detJ("detJ", nelems);
  View3D<T> J("J", nelems, 2, 2);
  View3D<T> invJ("invJ", nelems, 2, 2);

  View1D<T> Nxi("Nxi", 4);
  View1D<T> Neta("Neta", 4);

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
          Be(n, 0, i * 2) = invJ(n, 0, 0) * Nxi(i) + invJ(n, 0, 1) * Neta(i);
          Be(n, 1, i * 2 + 1) = invJ(n, 1, 0) * Nxi(i) + invJ(n, 1, 1) * Neta(i);
          Be(n, 2, i * 2) = Be(n, 1, i * 2 + 1);
          Be(n, 2, i * 2 + 1) = Be(n, 0, i * 2);
        }
      });

  return detJ;
}

template <typename T, typename D>
View3D<D> computeK(const View1D<D>& rho, const View3D<T>& detJ, const View5D<T>& Be,
                   const View2D<int>& conn, const View2D<T>& C0, const T rho0_K,
                   const char* ptype_K, const double p, const double q) {
  const int nelems = conn.extent(0);

  View3D<D> C("C", nelems, 3, 3);
  View3D<D> Ke("Ke", nelems, 8, 8);

  // Prepare the shape functions
  Kokkos::parallel_for(
      nelems, KOKKOS_LAMBDA(const int n) {
        // Average the density to get the element - wise density
        D rhoE_n = 0.25 * (rho(conn(n, 0)) + rho(conn(n, 1)) + rho(conn(n, 2)) + rho(conn(n, 3)));

        // Compute the constitutivve matrix
        const char* simp = "simp";
        const char* ramp = "ramp";
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            if (strcmp(ptype_K, simp) == 0) {
              C(n, i, j) = (std::pow(rhoE_n, p) + rho0_K) * C0(i, j);
            } else if (strcmp(ptype_K, ramp) == 0) {
              C(n, i, j) = (rhoE_n / (1.0 + q * (1.0 - rhoE_n))) * C0(i, j);
            } else {
              printf("Penalty type not supported\n");
            }
          }
        }
      });

  for (int jj = 0; jj < 2; jj++) {
    for (int ii = 0; ii < 2; ii++) {
      auto detJ_ij = Kokkos::subview(detJ, ii, jj, Kokkos::ALL());
      auto Be_ij = Kokkos::subview(Be, ii, jj, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());

      // Ke += np.einsum("n,nij,nik,nkl -> njl", detJ, Be, C, Be)
      Kokkos::parallel_for(
          nelems, KOKKOS_LAMBDA(int n) {
            for (int j = 0; j < 8; j++) {
              for (int l = 0; l < 8; l++) {
                D temp = 0.0;
                for (int i = 0; i < 3; i++) {
                  for (int k = 0; k < 3; k++) {
                    temp += detJ_ij(n) * Be_ij(n, i, j) * C(n, i, k) * Be_ij(n, k, l);
                  }
                }
                Ke(n, j, l) += temp;
              }
            }
          });
    };
  };

  return Ke;
}

/*
  Compute the derivative of the stiffness matrix times the vectors psi and u
*/
template <typename T>
View1D<T> computeKDerivative(const View1D<T>& rho, const View3D<T>& detJ, const View5D<T>& Be,
                             const View2D<int>& conn, const View1D<T>& u, const View1D<T>& psi,
                             const View2D<T>& C0, const char* ptype_K, const double p,
                             const double q) {
  const int nelems = conn.extent(0);
  const int nnodes = rho.extent(0);

  View2D<T> ue("ue", nelems, 8);
  View2D<T> psie("psie", nelems, 8);
  View3D<T> dfdC("dfdC", nelems, 3, 3);
  View1D<T> dK("dK", nnodes);

  // Prepare the shape functions
  Kokkos::parallel_for(
      nelems, KOKKOS_LAMBDA(const int n) {
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 2; ++j) {
            ue(n, i * 2 + j) = u(conn(n, i) * 2 + j);
            psie(n, i * 2 + j) = psi(conn(n, i) * 2 + j);
          }
        }
      });

  for (int jj = 0; jj < 2; jj++) {
    for (int ii = 0; ii < 2; ii++) {
      auto detJ_ij = Kokkos::subview(detJ, ii, jj, Kokkos::ALL());
      auto Be_ij = Kokkos::subview(Be, ii, jj, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());

      // do not use MDRangePolicy and atomic_add
      Kokkos::parallel_for(
          nelems, KOKKOS_LAMBDA(int n) {
            for (int i = 0; i < 3; i++) {
              for (int j = 0; j < 3; j++) {
                T temp = 0.0;
                for (int m = 0; m < 8; m++) {
                  for (int l = 0; l < 8; l++) {
                    temp += detJ_ij(n) * Be_ij(n, i, m) * Be_ij(n, j, l) * psie(n, m) * ue(n, l);
                  }
                }
                dfdC(n, i, j) += temp;
              }
            }
          });
    };
  };

  Kokkos::parallel_for(
      nelems, KOKKOS_LAMBDA(const int n) {
        T drhoE_n = 0.0;
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            drhoE_n += dfdC(n, i, j) * C0(i, j);
          }
        }

        // Average the density to get the element - wise density
        T rhoE_n = 0.25 * (rho(conn(n, 0)) + rho(conn(n, 1)) + rho(conn(n, 2)) + rho(conn(n, 3)));

        // Penalize the stiffness matrix
        const char* simp = "simp";
        const char* ramp = "ramp";
        if (strcmp(ptype_K, simp) == 0) {
          drhoE_n *= p * pow(rhoE_n, p - 1.0);
        } else if (strcmp(ptype_K, ramp) == 0) {
          drhoE_n *= (1.0 + q) / pow((1.0 + q * (1.0 - rhoE_n)), 2.0);
        } else {
          printf("Penalty type not supported\n");
        }

        for (int i = 0; i < 4; i++) {
          Kokkos::atomic_add(&dK(conn(n, i)), 0.25 * drhoE_n);
        }
      });

  return dK;
}

#endif  // STIFFNESS_H

// template <typename T, typename D>
// View3D<D> computeK(const View2D<T>& X, const View2D<int>& conn, const View1D<D>& rho,
//                    const View2D<T>& C0, const T rho0_K, const char* ptype_K, const double p,
//                    const double q) {
//   const int nelems = conn.extent(0);
//   const int nnodes = X.extent(0);

//   View3D<D> C("C", nelems, 3, 3);
//   View2D<T> xe("xe", nelems, 4);
//   View2D<T> ye("ye", nelems, 4);
//   View3D<T> Be("Be", nelems, 3, 8);
//   View3D<D> Ke("Ke", nelems, 8, 8);

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
//           for (int j = 0; j < 3; j++) {
//             if (strcmp(ptype_K, simp) == 0) {
//               C(n, i, j) = (std::pow(rhoE_n, p) + rho0_K) * C0(i, j);
//             } else if (strcmp(ptype_K, ramp) == 0) {
//               C(n, i, j) = (rhoE_n / (1.0 + q * (1.0 - rhoE_n))) * C0(i, j);
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
//         }
//       });

//   for (int jj = 0; jj < 2; jj++) {
//     for (int ii = 0; ii < 2; ii++) {
//       T xi = gauss_pts[ii];
//       T eta = gauss_pts[jj];

//       auto detJ = populateBe(xi, eta, xe, ye, Be);

//       // Ke += np.einsum("n,nij,nik,nkl -> njl", detJ, Be, C, Be)
//       Kokkos::parallel_for(
//           Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {8, 8, nelems}),
//           KOKKOS_LAMBDA(int j, int l, int n) {
//             for (int i = 0; i < 3; i++) {
//               for (int k = 0; k < 3; k++) {
//                 Ke(n, j, l) += detJ(n) * Be(n, i, j) * C(n, i, k) * Be(n, k, l);
//               }
//             }
//           });
//     };
//   };

//   return Ke;
// }

// /*
//   Compute the derivative of the stiffness matrix times the vectors psi and u
// */
// template <typename T>
// View1D<T> computeKDerivative(const View2D<T>& X, const View2D<int>& conn, const View1D<T>& rho,
//                              const View1D<T>& u, const View1D<T>& psi, const View2D<T>& C0,
//                              const char* ptype_K, const double p, const double q) {
//   const int nelems = conn.extent(0);
//   const int nnodes = X.extent(0);

//   View2D<T> xe("xe", nelems, 4);
//   View2D<T> ye("ye", nelems, 4);
//   View2D<T> ue("ue", nelems, 8);
//   View2D<T> psie("psie", nelems, 8);
//   View3D<T> Be("Be", nelems, 3, 8);
//   View3D<T> dfdC("dfdC", nelems, 3, 3);
//   View1D<T> dK("dK", nnodes);

//   // Compute Gauss quadrature with a 2-point quadrature rule
//   const double gauss_pts[2] = {-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)};

//   // Prepare the shape functions
//   Kokkos::parallel_for(
//       nelems, KOKKOS_LAMBDA(const int n) {
//         for (int i = 0; i < 4; ++i) {
//           xe(n, i) = X(conn(n, i), 0);
//           ye(n, i) = X(conn(n, i), 1);
//           ue(n, i * 2) = u(conn(n, i) * 2);
//           ue(n, i * 2 + 1) = u(conn(n, i) * 2 + 1);
//           psie(n, i * 2) = psi(conn(n, i) * 2);
//           psie(n, i * 2 + 1) = psi(conn(n, i) * 2 + 1);
//         }
//       });

//   for (int jj = 0; jj < 2; jj++) {
//     for (int ii = 0; ii < 2; ii++) {
//       T xi = gauss_pts[ii];
//       T eta = gauss_pts[jj];

//       auto detJ = populateBe(xi, eta, xe, ye, Be);

//       // dfdC += np.einsum("n,nim,njl,nm,nl -> nij", detJ, Be, Be, psie, ue)
//       Kokkos::parallel_for(
//           Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {3, 3, nelems}),
//           KOKKOS_LAMBDA(int i, int j, int n) {
//             for (int m = 0; m < 8; m++) {
//               for (int l = 0; l < 8; l++) {
//                 dfdC(n, i, j) += detJ(n) * Be(n, i, m) * Be(n, j, l) * psie(n, m) * ue(n, l);
//               }
//             }
//           });
//     };
//   };

//   Kokkos::parallel_for(
//       nelems, KOKKOS_LAMBDA(const int n) {
//         T drhoE_n = 0.0;
//         for (int i = 0; i < 3; i++) {
//           for (int j = 0; j < 3; j++) {
//             drhoE_n += dfdC(n, i, j) * C0(i, j);
//           }
//         }

//         // Average the density to get the element - wise density
//         T rhoE_n = 0.25 * (rho(conn(n, 0)) + rho(conn(n, 1)) + rho(conn(n, 2)) + rho(conn(n,
//         3)));

//         // Penalize the stiffness matrix
//         const char* simp = "simp";
//         const char* ramp = "ramp";
//         if (strcmp(ptype_K, simp) == 0) {
//           drhoE_n *= p * pow(rhoE_n, p - 1.0);
//         } else if (strcmp(ptype_K, ramp) == 0) {
//           drhoE_n *= (1.0 + q) / pow((1.0 + q * (1.0 - rhoE_n)), 2.0);
//         } else {
//           printf("Penalty type not supported\n");
//         }

//         for (int i = 0; i < 4; i++) {
//           Kokkos::atomic_add(&dK(conn(n, i)), 0.25 * drhoE_n);
//         }
//       });

//   return dK;
// }