#pragma once
#ifndef LOBPCG_SPARSE_HPP
#define LOBPCG_SPARSE_HPP

#include <KokkosBlas.hpp>
#include <KokkosBlas1_iamax.hpp>
#include <KokkosSparse_gmres.hpp>
#include <cstdlib>

#include "KokkosSparse_spgemm.hpp"
#include "lapackage.hpp"
#include "utils.hpp"

namespace linalg::sparse {
/**
 * Given a real symmetric 3x3 matrix A, compute selected eigenvalues and eigenvectors use analytical
 * method
 *
 * Input:
 *   Ap: real symmetric 3x3 matrix A
 *    m: first m eigenvalues and eigenvectors to be computed
 *
 * Output:
 *   wp: first m eigenvalues of A
 *   vp: first m eigenvectors of A
 *
 * Note:
 *    1. use analytical method
 *    2. eigenvalues and eigenvectors are sorted in ascending order
 *    3. 'A' have to be normalised, otherwise the result will lose precision for case ¦Ëi >> ¦Ëj
 *
 * Reference:
 *    1. https://en.wikipedia.org/wiki/Eigenvalue_algorithm#cite_note-Smith-19
 *    2. Smith, Oliver K. (April 1961), "Eigenvalues of a symmetric 3 ¡Á 3 matrix.", Communications
 * of the ACM, 4 (4): 168, doi:10.1145/355578.366316, S2CID 37815415
 */
template <typename T>
void syevx3x3_analytical(T Ap[], int m, T wp[], T vp[], bool verbose = true) {
  constexpr auto pi = Kokkos::numbers::pi_v<T>;
  HostView2D<T> A(Ap, 3, 3);  // RowMajor or ColMajor are the same since A is symmetric
  HostView1D<T> w(wp, m);
  HostView2D<T> v(vp, 3, m);  // RowMajor for host, ColMajor for device

  T a00 = A(0, 0);
  T a01 = A(0, 1);
  T a02 = A(0, 2);
  T a11 = A(1, 1);
  T a12 = A(1, 2);
  T a22 = A(2, 2);

  T p1 = a01 * a01 + a02 * a02 + a12 * a12;

  /* Check if matrix is diagonal */
  if (p1 == 0) {
    for (int i = 0; i < m; ++i) {
      w(i) = A(i, i);  // eigenvalues are diagonal elements
      for (int j = 0; j < m; ++j) {
        v(j, i) = (i == j) ? 1 : 0;  // eigenvectors are the identity matrix
      }
    }
    return;
  }

  T q = (a00 + a11 + a22) / 3;  // trace(A) / 3

  T b00 = a00 - q;
  T b11 = a11 - q;
  T b22 = a22 - q;

  T p = sqrt((b00 * b00 + b11 * b11 + b22 * b22 + 2 * p1) / 6);  // norm(A - q * I) / sqrt(6)

  /* Compute the determinant of B */
  T detB = (b00 * (b11 * b22 - a12 * a12) - a01 * (a01 * b22 - a12 * a02) +
            a02 * (a01 * a12 - b11 * a02));

  T r = detB / (2 * p * p * p);

  // In exact arithmetic for a symmetric matrix  -1 <= r <= 1
  // but computation error can leave it slightly outside this range.
  T phi;
  if (r <= -1)
    phi = pi / 3;
  else if (r >= 1)
    phi = 0;
  else
    phi = Kokkos::acos(r) / 3;

  /* Compute eigenvalues, the eigenvalues satisfy ¦Ë0 <= ¦Ë1 <= ¦Ë2 */
  T w0 = q + 2 * p * Kokkos::cos(phi + (2 * pi / 3));
  T w1 = q + 2 * p * Kokkos::cos(phi - (2 * pi / 3));
  T w2 = 3 * q - w0 - w1;  // since trace(A) = eig1 + eig2 + eig3

  /* Compute eigenvectors */
  /* v[:, 0] = (A - w(1) * I) * (A - w(2) * I)[: , 1] */
  v(0, 0) = (a00 - w1) * a01 + a01 * (a11 - w2) + a02 * a12;
  v(1, 0) = a01 * a01 + (a11 - w1) * (a11 - w2) + a12 * a12;
  v(2, 0) = a02 * a01 + a12 * (a11 - w2) + (a22 - w1) * a12;

  T norm1 = sqrt(v(0, 0) * v(0, 0) + v(1, 0) * v(1, 0) + v(2, 0) * v(2, 0));

  w(0) = w0;
  v(0, 0) /= norm1;
  v(1, 0) /= norm1;
  v(2, 0) /= norm1;

  /* v[:, 1] = (A - w(2) * I) * (A - w(0) * I)[: , 2] */
  if (m > 1) {
    v(0, 1) = (a00 - w2) * a02 + a01 * a12 + a02 * (a22 - w0);
    v(1, 1) = a01 * a02 + (a11 - w2) * a12 + a12 * (a22 - w0);
    v(2, 1) = a02 * a02 + a12 * a12 + (a22 - w2) * (a22 - w0);

    T norm2 = sqrt(v(0, 1) * v(0, 1) + v(1, 1) * v(1, 1) + v(2, 1) * v(2, 1));

    w(1) = w1;
    v(0, 1) /= norm2;
    v(1, 1) /= norm2;
    v(2, 1) /= norm2;
  }

  /* v[:, 2] = (A - w(0) * I) * (A - w(1) * I)[: , 0] */
  if (m > 2) {
    v(0, 2) = (a00 - w0) * (a00 - w1) + a01 * a01 + a02 * a02;
    v(1, 2) = a01 * (a00 - w1) + (a11 - w0) * a01 + a12 * a02;
    v(2, 2) = a02 * (a00 - w1) + a12 * a01 + (a22 - w0) * a02;

    T norm3 = sqrt(v(0, 2) * v(0, 2) + v(1, 2) * v(1, 2) + v(2, 2) * v(2, 2));

    w(2) = w2;
    v(0, 2) /= norm3;
    v(1, 2) /= norm3;
    v(2, 2) /= norm3;
  }
}

/**
 * Computes selected eigenpairs for 3x3 real generalized symmetric-definite eigenproblem Ax=¦ËBx
 *
 * Input:
 *   Ap: pointer for real symmetric 3x3 matrix A
 *   Bp: pointer for real symmetric 3x3 matrix B
 *    m: first m eigenvalues and eigenvectors to be computed
 *
 * Output:
 *   wp: first m eigenvalues of A
 *   vp: first m eigenvectors of A
 *
 * Note:
 *    1. Algorithm 1 in reference 1
 *    2. eigenvalues and eigenvectors are sorted in ascending order
 *    3. 'A' have to be normalised, otherwise the result will lose precision for case ¦Ëi >> ¦Ëj
 *
 * Algorithm:
 *    1. ¦µB, ¦«B <- B * ¦µB = ¦µB * ¦«B
 *    2. ¦µB_hat <- ¦µB_hat = ¦µB * ¦«B^(?1/2) ¡Ö ¦µB * (¦«B^(1/2) + ¦ÅI)^(?1)
 *    3. A_hat <- A_hat = ¦µB_hat * A * ¦µB_hat
 *    4. ¦µA, ¦«A <- A_hat * ¦µA = ¦µA * ¦«A
 *    5. ¦« <- ¦«A, ¦µ <- ¦µB_hat * ¦µA
 *
 * Reference:
 *    1. Ghojogh B, Karray F, Crowley M. Eigenvalue and generalized eigenvalue problems:
 * Tutorial[J]. arXiv preprint arXiv:1903.11240, 2019.
 */
template <typename T>
void sygvx3x3(T Ap[], T Bp[], int m, T wp[], T vp[], bool verbose = true) {
  HostView2D<T> A(Ap, 3, 3);
  HostView2D<T> B(Bp, 3, 3);

  /* Compute eigenvalues and eigenvectors of B */
  HostView2D<T> vB("eigenvectors of B", 3, 3);
  HostView1D<T> wB("eigenvalues of B", 3);
  syevx3x3_analytical(B.data(), 3, wB.data(), vB.data());

  /* Compute ¦µB_hat = ¦µB * (¦«B^(1/2) + ¦ÅI)^(?1), in case ¦«B^(1/2) is singular */
  T eps = std::numeric_limits<T>::epsilon();
  wB(0) = 1 / (sqrt(wB(0)) + eps);
  wB(1) = 1 / (sqrt(wB(1)) + eps);
  wB(2) = 1 / (sqrt(wB(2)) + eps);

  vB(0, 0) *= wB(0);
  vB(1, 0) *= wB(0);
  vB(2, 0) *= wB(0);

  vB(0, 1) *= wB(1);
  vB(1, 1) *= wB(1);
  vB(2, 1) *= wB(1);

  vB(0, 2) *= wB(2);
  vB(1, 2) *= wB(2);
  vB(2, 2) *= wB(2);

  /* Compute A_hat = ¦µB_hat * A * ¦µB_hat */
  HostView2D<T> A_hat("A_hat", 3, 3);
  T a00 = A(0, 0) * vB(0, 0) + A(0, 1) * vB(1, 0) + A(0, 2) * vB(2, 0);
  T a10 = A(1, 0) * vB(0, 0) + A(1, 1) * vB(1, 0) + A(1, 2) * vB(2, 0);
  T a20 = A(2, 0) * vB(0, 0) + A(2, 1) * vB(1, 0) + A(2, 2) * vB(2, 0);

  T a01 = A(0, 0) * vB(0, 1) + A(0, 1) * vB(1, 1) + A(0, 2) * vB(2, 1);
  T a11 = A(1, 0) * vB(0, 1) + A(1, 1) * vB(1, 1) + A(1, 2) * vB(2, 1);
  T a21 = A(2, 0) * vB(0, 1) + A(2, 1) * vB(1, 1) + A(2, 2) * vB(2, 1);

  T a02 = A(0, 0) * vB(0, 2) + A(0, 1) * vB(1, 2) + A(0, 2) * vB(2, 2);
  T a12 = A(1, 0) * vB(0, 2) + A(1, 1) * vB(1, 2) + A(1, 2) * vB(2, 2);
  T a22 = A(2, 0) * vB(0, 2) + A(2, 1) * vB(1, 2) + A(2, 2) * vB(2, 2);

  A_hat(0, 0) = vB(0, 0) * a00 + vB(1, 0) * a10 + vB(2, 0) * a20;
  A_hat(0, 1) = vB(0, 0) * a01 + vB(1, 0) * a11 + vB(2, 0) * a21;
  A_hat(0, 2) = vB(0, 0) * a02 + vB(1, 0) * a12 + vB(2, 0) * a22;
  A_hat(1, 1) = vB(0, 1) * a01 + vB(1, 1) * a11 + vB(2, 1) * a21;
  A_hat(1, 2) = vB(0, 1) * a02 + vB(1, 1) * a12 + vB(2, 1) * a22;
  A_hat(2, 2) = vB(0, 2) * a02 + vB(1, 2) * a12 + vB(2, 2) * a22;

  A_hat(1, 0) = A_hat(0, 1);
  A_hat(2, 0) = A_hat(0, 2);
  A_hat(2, 1) = A_hat(1, 2);

  /* Compute first m eigenpair of A_hat */
  HostView2D<T> vA("eigenvectors of A_hat", 3, m);
  syevx3x3_analytical(A_hat.data(), m, wp, vA.data());

  /* Compute eigenvectors ¦µ <- ¦µB_hat * ¦µA */
  HostView2D<T> v(vp, 3, m);

  v(0, 0) = vB(0, 0) * vA(0, 0) + vB(0, 1) * vA(1, 0) + vB(0, 2) * vA(2, 0);
  v(1, 0) = vB(1, 0) * vA(0, 0) + vB(1, 1) * vA(1, 0) + vB(1, 2) * vA(2, 0);
  v(2, 0) = vB(2, 0) * vA(0, 0) + vB(2, 1) * vA(1, 0) + vB(2, 2) * vA(2, 0);

  if (m > 1) {
    v(0, 1) = vB(0, 0) * vA(0, 1) + vB(0, 1) * vA(1, 1) + vB(0, 2) * vA(2, 1);
    v(1, 1) = vB(1, 0) * vA(0, 1) + vB(1, 1) * vA(1, 1) + vB(1, 2) * vA(2, 1);
    v(2, 1) = vB(2, 0) * vA(0, 1) + vB(2, 1) * vA(1, 1) + vB(2, 2) * vA(2, 1);
  }

  if (m > 2) {
    v(0, 2) = vB(0, 0) * vA(0, 2) + vB(0, 1) * vA(1, 2) + vB(0, 2) * vA(2, 2);
    v(1, 2) = vB(1, 0) * vA(0, 2) + vB(1, 1) * vA(1, 2) + vB(1, 2) * vA(2, 2);
    v(2, 2) = vB(2, 0) * vA(0, 2) + vB(2, 1) * vA(1, 2) + vB(2, 2) * vA(2, 2);
  }
}

/**
 * Given a real symmetric 2x2 matrix A, compute selected eigenvalues and eigenvectors use analytical
 * method
 */
template <typename T>
void syevx2x2_analytical(T Ap[], int m, T wp[], T vp[], bool verbose = true) {
  constexpr auto pi = Kokkos::numbers::pi_v<T>;
  HostView2D<T> A(Ap, 2, 2);  // RowMajor or ColMajor are the same since A is symmetric
  HostView1D<T> w(wp, m);
  HostView2D<T> v(vp, 2, m);  // RowMajor for host, ColMajor for device

  T a00 = A(0, 0);
  T a01 = A(0, 1);
  T a11 = A(1, 1);

  /* Check if matrix is diagonal */
  if (a01 * a01 == 0) {
    for (int i = 0; i < m; ++i) {
      w(i) = A(i, i);  // eigenvalues are diagonal elements
      for (int j = 0; j < m; ++j) {
        v(j, i) = (i == j) ? 1 : 0;  // eigenvectors are the identity matrix
      }
    }
    return;
  }

  /* Compute eigenvalues, the eigenvalues satisfy ¦Ë0 <= ¦Ë1 */
  T trA = a00 + a11;
  T detA = a00 * a11 - a01 * a01;
  T gapA = sqrt(trA * trA - 4 * detA);

  T w0 = (trA - gapA) / 2;
  T w1 = (trA + gapA) / 2;

  /* Compute eigenvectors */
  v(0, 0) = 1 / sqrt(1 + (w0 - a00) * (w0 - a00) / (a01 * a01));
  v(1, 0) = v(0, 0) * (w0 - a00) / a01;
  w(0) = w0;

  if (m > 1) {
    v(0, 1) = 1 / sqrt(1 + (w1 - a00) * (w1 - a00) / (a01 * a01));
    v(1, 1) = v(0, 1) * (w1 - a00) / a01;
    w(1) = w1;
  }
}

/**
 * Computes selected eigenpairs for 2x2 real generalized symmetric-definite eigenproblem Ax=¦ËBx
 */
template <typename T>
void sygvx2x2(T Ap[], T Bp[], int m, T wp[], T vp[], bool verbose = true) {
  HostView2D<T> A(Ap, 2, 2);
  HostView2D<T> B(Bp, 2, 2);

  /* Compute eigenvalues and eigenvectors of B */
  HostView2D<T> vB("eigenvectors of B", 2, 2);
  HostView1D<T> wB("eigenvalues of B", 2);
  syevx2x2_analytical(B.data(), 2, wB.data(), vB.data());

  /* Compute ¦µB_hat = ¦µB * (¦«B^(1/2) + ¦ÅI)^(?1), in case ¦«B^(1/2) is singular */
  T eps = std::numeric_limits<T>::epsilon();
  wB(0) = 1 / (sqrt(wB(0)) + eps);
  wB(1) = 1 / (sqrt(wB(1)) + eps);

  vB(0, 0) *= wB(0);
  vB(1, 0) *= wB(0);

  vB(0, 1) *= wB(1);
  vB(1, 1) *= wB(1);

  /* Compute A_hat = ¦µB_hat * A * ¦µB_hat */
  HostView2D<T> A_hat("A_hat", 2, 2);
  T a00 = A(0, 0) * vB(0, 0) + A(0, 1) * vB(1, 0);
  T a10 = A(1, 0) * vB(0, 0) + A(1, 1) * vB(1, 0);

  T a01 = A(0, 0) * vB(0, 1) + A(0, 1) * vB(1, 1);
  T a11 = A(1, 0) * vB(0, 1) + A(1, 1) * vB(1, 1);

  A_hat(0, 0) = vB(0, 0) * a00 + vB(1, 0) * a10;
  A_hat(0, 1) = vB(0, 0) * a01 + vB(1, 0) * a11;
  A_hat(1, 1) = vB(0, 1) * a01 + vB(1, 1) * a11;
  A_hat(1, 0) = A_hat(0, 1);

  /* Compute first m eigenpair of A_hat */
  HostView2D<T> vA("eigenvectors of A_hat", 2, m);
  syevx2x2_analytical(A_hat.data(), m, wp, vA.data());

  /* Compute eigenvectors ¦µ <- ¦µB_hat * ¦µA */
  HostView2D<T> v(vp, 2, m);

  v(0, 0) = vB(0, 0) * vA(0, 0) + vB(0, 1) * vA(1, 0);
  v(1, 0) = vB(1, 0) * vA(0, 0) + vB(1, 1) * vA(1, 0);

  if (m > 1) {
    v(0, 1) = vB(0, 0) * vA(0, 1) + vB(0, 1) * vA(1, 1);
    v(1, 1) = vB(1, 0) * vA(0, 1) + vB(1, 1) * vA(1, 1);
  }
}

template <typename T>
void sygvx3x3(T Ap[], T Bp[], int n, int m, T wp[], T vp[], bool verbose = true) {
  if (n == 2) {
    sygvx2x2<T>(Ap, Bp, m, wp, vp, verbose);
  } else if (n == 3) {
    sygvx3x3<T>(Ap, Bp, m, wp, vp, verbose);
  } else {
    printf("sygvx3x3: n = %d is not supported\n", n);
  }
}

/* *
 * Compute residual R = AX - BX * w, R_ = AX + BX * w
 */
template <typename T>
void compute_residual(const View2D<T>& AX, const View2D<T>& BX, const View1D<T>& w, const int n,
                      const int m, View2D<T>& R, View2D<T>& R_) {
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, m}),
      KOKKOS_LAMBDA(const int i, const int j) {
        R(i, j) = AX(i, j) - BX(i, j) * w(j);
        R_(i, j) = AX(i, j) + BX(i, j) * w(j);
      });
}

/* *
 * Compute norm of columns of X, R, P, R_
 */
template <typename T>
void compute_norm(const View2D<T>& X, const View2D<T>& R, const View2D<T>& P, const View2D<T>& R_,
                  const View1D<T>& is_convergent, const int n, const int m, const int m0,
                  View2D<T>& norm) {
  KokkosBlas::fill(norm, 0.0);  // initialize norm to 0

  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, m}),
      KOKKOS_LAMBDA(const int i, const int j) {
        Kokkos::atomic_add(&norm(j, 0), X(i, j) * X(i, j));
        Kokkos::atomic_add(&norm(j, 1), R(i, j) * R(i, j));
        Kokkos::atomic_add(&norm(j, 2), P(i, j) * P(i, j));

        if (j < m0) {
          Kokkos::atomic_add(&norm(j, 3), R_(i, j) * R_(i, j));
        }
      });
}

/* *
 * Compute norm of columns of X, R, and ||P||^2 = 1
 */
template <typename T>
void compute_norm(const View2D<T>& X, const View2D<T>& R, const int n, const int m,
                  View2D<T>& norm) {
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, m}),
      KOKKOS_LAMBDA(const int i, const int j) {
        Kokkos::atomic_add(&norm(j, 0), X(i, j) * X(i, j));  // norm(j, 0) = ||Xi||^2
        Kokkos::atomic_add(&norm(j, 1), R(i, j) * R(i, j));  // norm(j, 1) = ||Ri||^2
        if (i == 0) {
          norm(j, 2) = 1.0;  // norm(j, 2) = 1.0, since Pi = 0
        }
      });
}

/* *
 * Compute residual nrom = ||R|| / ||R_|| = ||AX - BX * w|| / ||AX + BX * w||
 */
template <typename T>
void compute_residual_norm(const View2D<T>& norm, const View1D<T>& is_convergent, const int m0,
                           View1D<T>& residual) {
  Kokkos::parallel_for(
      m0, KOKKOS_LAMBDA(const int i) {
        if (is_convergent(i) == 0) {
          residual(i) = sqrt(norm(i, 1) / norm(i, 3));  // res(i) = ||Ri|| / ||Ri_||
        }
      });
}

/**
 * Checks convergence for lobpcg I and II
 */
template <typename T>
bool check_convergence(T residual[], T is_convergent[], const int m, const int k, const int maxIter,
                       const double tol, bool verbose = true) {
  T max_residual = 0.0;
  int count = 0;
  bool converged = false;

  for (int i = 0; i < m; i++) {
    max_residual = std::max(max_residual, residual[i]);

    if (residual[i] < tol) {
      is_convergent[i] = 1.0;
      count++;
    } else {
      is_convergent[i] = 0.0;
    }
  }

  if (verbose) {
    printf(
        "Iteration: \033[32m%2d\033[0m, converged: \033[32m%2d\033[0m, "
        "residual: \033[32m%e\033[0m\n",
        k, count, max_residual);
  }

  if (max_residual < tol || count == m) {
    converged = true;
  }

  if (k == maxIter - 1) {
    printf(
        "\033[1;31mWarning\033[0m: maximum number of iterations reached, "
        "residual: %e\n",
        max_residual);
  }

  return converged;
}

/* *
 * Compute gram matrix: gramA = S^T * A * S, gramB = S^T * B * S
 */
template <typename T>
void compute_gramAB(const View2D<T>& X, const View2D<T>& AX, const View2D<T>& BX,
                    const View2D<T>& W, const View2D<T>& AW, const View2D<T>& BW,
                    const View2D<T>& P, const View2D<T>& AP, const View2D<T>& BP, const int m,
                    View2D<T>& gramA, View2D<T>& gramB) {
  auto pair_0_m = Kokkos::make_pair(0, m);
  auto pair_m_2m = Kokkos::make_pair(m, 2 * m);
  auto pair_2m_3m = Kokkos::make_pair(2 * m, 3 * m);

  auto XAX = Kokkos::subview(gramA, pair_0_m, pair_0_m);
  auto XBX = Kokkos::subview(gramB, pair_0_m, pair_0_m);
  auto XAW = Kokkos::subview(gramA, pair_0_m, pair_m_2m);
  auto XBW = Kokkos::subview(gramB, pair_0_m, pair_m_2m);
  auto XAP = Kokkos::subview(gramA, pair_0_m, pair_2m_3m);
  auto XBP = Kokkos::subview(gramB, pair_0_m, pair_2m_3m);
  auto WAW = Kokkos::subview(gramA, pair_m_2m, pair_m_2m);
  auto WBW = Kokkos::subview(gramB, pair_m_2m, pair_m_2m);
  auto WAP = Kokkos::subview(gramA, pair_m_2m, pair_2m_3m);
  auto WBP = Kokkos::subview(gramB, pair_m_2m, pair_2m_3m);
  auto PAP = Kokkos::subview(gramA, pair_2m_3m, pair_2m_3m);
  auto PBP = Kokkos::subview(gramB, pair_2m_3m, pair_2m_3m);

  KokkosBlas::gemm("T", "N", 1.0, X, AX, 0.0, XAX);  // XAX = X^T * AX
  KokkosBlas::gemm("T", "N", 1.0, X, BX, 0.0, XBX);  // XBX = X^T * BX
  KokkosBlas::gemm("T", "N", 1.0, X, AW, 0.0, XAW);  // XAW = X^T * AW
  KokkosBlas::gemm("T", "N", 1.0, X, BW, 0.0, XBW);  // XBW = X^T * BW
  KokkosBlas::gemm("T", "N", 1.0, X, AP, 0.0, XAP);  // XAP = X^T * AP
  KokkosBlas::gemm("T", "N", 1.0, X, BP, 0.0, XBP);  // XBP = X^T * BP
  KokkosBlas::gemm("T", "N", 1.0, W, AW, 0.0, WAW);  // WAW = W^T * AW
  KokkosBlas::gemm("T", "N", 1.0, W, BW, 0.0, WBW);  // WBW = W^T * BW
  KokkosBlas::gemm("T", "N", 1.0, W, AP, 0.0, WAP);  // WAP = W^T * AP
  KokkosBlas::gemm("T", "N", 1.0, W, BP, 0.0, WBP);  // WBP = W^T * BP
  KokkosBlas::gemm("T", "N", 1.0, P, AP, 0.0, PAP);  // PAP = P^T * AP
  KokkosBlas::gemm("T", "N", 1.0, P, BP, 0.0, PBP);  // PBP = P^T * BP

  Kokkos::parallel_for(
      "update_gram", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m, m}),
      KOKKOS_LAMBDA(const int i, const int j) {
        gramA(i + m, j) = XAW(j, i);
        gramB(i + m, j) = XBW(j, i);
        gramA(i + 2 * m, j) = XAP(j, i);
        gramB(i + 2 * m, j) = XBP(j, i);
        gramA(i + 2 * m, j + m) = WAP(j, i);
        gramB(i + 2 * m, j + m) = WBP(j, i);
      });
}

/**
 * convert dense matrix to CRS format
 *
 * Input:
 *  n: number of rows
 *  m: number of columns
 *
 * Output:
 * Ap: pointer for row pointer
 * Aj: pointer for column index
 *
 * Note:
 *   ignore zero elements
 *   Ap = [0, m, 2m, ..., nm]
 *   Aj = [0, 1, 2, ..., m-1, 0, 1, 2, ..., m-1, ..., 0, 1, 2, ..., m-1]
 */
template <typename I>
void denseToCrs(int n, int m, I Ap[], I Aj[]) {
  for (int i = 0; i < n; ++i) {
    Ap[i] = i * m;
    for (int j = 0; j < m; ++j) {
      Aj[i * m + j] = j;
    }
  }
  Ap[n] = n * m;
}

/**
 * convert CRS matrix to dense format for first n rows and m columns
 *
 * Dtype:
 *  T: value type
 *  I: index type
 *
 * Input:
 *  Ax: pointer for values
 *  Ap: pointer for row pointer
 *  Aj: pointer for column index
 *   n: number of rows
 *   m: number of columns
 *
 * Output:
 *   A: pointer for dense matrix
 */
template <typename T, typename I>
void csrPartialToDense(const T Ax[], const I Ap[], const I Aj[], const int n, const int m, T A[]) {
  Kokkos::parallel_for(
      "csr_to_dense", n, KOKKOS_LAMBDA(int i) {
        int startIdx = Ap[i];
        int endIdx = Ap[i + 1];

        for (int k = startIdx; k < endIdx; ++k) {
          int j = Aj[k];

          if (j < m) {
            A[i * m + j] = Ax[k];
          }
        }
      });
}

template <typename T, typename I>
void lobpcgII(T Ax[], I Ap[], I Aj[], T Bx[], I Bp[], I Bj[], int n, int m, T wp[], T vp[],
              T Xp[] = nullptr, T Mp[] = nullptr, double tol = 1e-8, int maxIter = 500,
              bool verbose = true) {
  // TODO: Bp is nullptr, use identity matrix
  // TODO: sygv for full matrix eigenvalue problem
  // TODO: if not converged, pick up the result for lowest residual
  // TODO: if the lowest residual is still above 1e-4, then fail
  // TODO: print residual precision based on tolerance set
  // TODO: Xp need to copy to device

  const int m0 = m;                    // number of eigenpairs desired
  const int m1 = int(ceil(2.0 * m0));  // added number of eigenpairs to compute
  m = m0 + m1;                         // total number of eigenpairs to compute
  m = n < m ? n : m;                   // m cannot be larger than n

  if (verbose) {
    printf(
        "lobpcg: n = \033[32m%d\033[0m, m = \033[32m%d\033[0m, m (added) = "
        "\033[32m%d\033[0m\n",
        n, m0, m - m0);
  }


  // make kernel handle and set the options for GMRES
  using EXSP = Kokkos::DefaultExecutionSpace;
  using MESP = typename EXSP::memory_space;
  using crsMat_t = KokkosSparse::CrsMatrix<T, I, EXSP, void, I>;
  using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<I, I, T, EXSP, MESP, MESP>;

  std::string alg("SPGEMM_KK_MEMSPEED");

  crsMat_t Acsr("Acrsmat", n, n, Ap[n], Ax, Ap, Aj);
  crsMat_t Bcsr("Bcrsmat", n, n, Bp[n], Bx, Bp, Bj);

  View1D<T> values("values", n * m);
  View1D<I> rowPtr("row pointer", n + 1);
  View1D<I> colInd("column index", n * m);

  denseToCrs<I>(n, m, rowPtr.data(), colInd.data());

  crsMat_t Pcsr("Pcrsmat", n, m, n * m, values.data(), rowPtr.data(), colInd.data());
  crsMat_t Wcsr("Wcrsmat", n, m, n * m, values.data(), rowPtr.data(), colInd.data());
  crsMat_t Xcsr;

  if (Xp == nullptr) {
    Xcsr = crsMat_t("Xcrsmat", n, m, n * m, values.data(), rowPtr.data(), colInd.data());
  } else {
    Xcsr = crsMat_t("Xcrsmat", n, m, n * m, Xp, rowPtr.data(), colInd.data());
  }

  View2D<T> A(Ax, n, n);
  View2D<T> B(Bx, n, n);

  /* store in vstack [ X | AX | BX ], [ P | AP | BP ], [ W | AW | BW ] */
  View2D<T> X_AX_BX("vstack: [ X | AX | BX ]", 3 * n, m);
  View2D<T> W_AW_BW("vstack: [ W | AW | BW ]", 3 * n, m);
  View2D<T> P_AP_BP("vstack: [ P | AP | BP ]", 3 * n, m);

  View2D<T> X = Kokkos::subview(X_AX_BX, Kokkos::make_pair(0, n), Kokkos::ALL());
  View2D<T> W = Kokkos::subview(W_AW_BW, Kokkos::make_pair(0, n), Kokkos::ALL());
  View2D<T> P = Kokkos::subview(P_AP_BP, Kokkos::make_pair(0, n), Kokkos::ALL());

  View2D<T> AX = Kokkos::subview(X_AX_BX, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());
  View2D<T> AW = Kokkos::subview(W_AW_BW, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());
  View2D<T> AP = Kokkos::subview(P_AP_BP, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());

  View2D<T> BX = Kokkos::subview(X_AX_BX, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());
  View2D<T> BW = Kokkos::subview(W_AW_BW, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());
  View2D<T> BP = Kokkos::subview(P_AP_BP, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());

  View2D<T> S("vstack: [Xi, Wi, Pi], m sub-blocks[n, 3]", m * n, 3);
  View2D<T> ABS("vstack: [AXi, AWi, APi, BXi, BWi, BPi], m sub-blocks[n, 6]", m * n, 6);

  View2D<T> tmp("temp for X_AX_BX", 3 * n, m);

  /* Compute XAX0 = X.T * A * X, XBX0 = X.T * B * X */
  View2D<T> XAX0("XAX0", m, m);
  View2D<T> XBX0("XBX0", m, m);

  if (Xp == nullptr) {
    csrPartialToDense<T, I>(Ax, Ap, Aj, m, m, XAX0.data());
    csrPartialToDense<T, I>(Bx, Bp, Bj, m, m, XBX0.data());
  } else {
    crsMat_t AXcsr;
    crsMat_t BXcsr;

    KernelHandle kh_AX;
    KernelHandle kh_BX;

    kh_AX.create_spgemm_handle(KokkosSparse::StringToSPGEMMAlgorithm(alg));
    kh_BX.create_spgemm_handle(KokkosSparse::StringToSPGEMMAlgorithm(alg));

    KokkosSparse::spgemm_symbolic(kh_AX, Acsr, 0, Xcsr, 0, AXcsr);
    KokkosSparse::spgemm_numeric(kh_AX, Acsr, 0, Xcsr, 0, AXcsr);

    KokkosSparse::spgemm_symbolic(kh_BX, Bcsr, 0, Xcsr, 0, BXcsr);
    KokkosSparse::spgemm_numeric(kh_BX, Bcsr, 0, Xcsr, 0, BXcsr);

    // check();

    csrPartialToDense<T, I>(AXcsr.values.data(), AXcsr.graph.row_map.data(),
                            AXcsr.graph.entries.data(), n, m, AX.data());
    csrPartialToDense<T, I>(BXcsr.values.data(), BXcsr.graph.row_map.data(),
                            BXcsr.graph.entries.data(), n, m, BX.data());

    kh_AX.destroy_spgemm_handle();
    kh_BX.destroy_spgemm_handle();

    X = View2D<T>(Xp, n, m);
    KokkosBlas::gemm("T", "N", 1.0, X, AX, 0.0, XAX0);  // XAX0 = X.T * AX
    KokkosBlas::gemm("T", "N", 1.0, X, BX, 0.0, XBX0);  // XBX0 = X.T * BX
  }

  // check();

  /* Solve generalized eigenvalue problem to get initial eigenvectors */
  View1D<T> w("eigenvalues", m);
  View2D<T> v("eigenvectors column major", m, m);

  lapackage::sygvx<T>(XAX0.data(), XBX0.data(), m, m, w.data(), v.data());

  // printmat("w", w.data(), 1, m);
  // printmat("v", v.data(), m, m);

  /* Compute: X = X * v, AX = A * X * v, BX = B * X * v */
  if (Xp == nullptr) {
    Kokkos::parallel_for(
        "X = eye(n, m) -> X = hstack(v, 0)", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m, m}),
        KOKKOS_LAMBDA(int i, int j) {
          X(i, j) = v(j, i);
          Xcsr.values[i * m + j] = v(j, i);
        });

    // check();

    crsMat_t AXcsr;
    crsMat_t BXcsr;

    // KokkosBlas::fill(AXcsr.values, 0.0);
    // KokkosBlas::fill(BXcsr.values, 0.0);

    KernelHandle kh_AX;
    KernelHandle kh_BX;

    kh_AX.create_spgemm_handle(KokkosSparse::StringToSPGEMMAlgorithm(alg));
    kh_BX.create_spgemm_handle(KokkosSparse::StringToSPGEMMAlgorithm(alg));

    // check();
    // printmat("Xcsr", Xcsr.values.data(), 1, 10 * m);
    // printmat("Acsr", Acsr.values.data(), 1, 10 * m);
    // printmat("Bcsr", Bcsr.values.data(), 1, 10 * m);

    KokkosSparse::spgemm_symbolic(kh_AX, Acsr, 0, Xcsr, 0, AXcsr);
    KokkosSparse::spgemm_symbolic(kh_BX, Bcsr, 0, Xcsr, 0, BXcsr);

    KokkosSparse::spgemm_numeric(kh_AX, Acsr, 0, Xcsr, 0, AXcsr);
    KokkosSparse::spgemm_numeric(kh_BX, Bcsr, 0, Xcsr, 0, BXcsr);

    // Kokkos::fence();
    // printmat("AXcsr", AXcsr.values.data(), 1, m);
    // printmat("BXcsr", BXcsr.values.data(), 1, m);

    // check();
    csrPartialToDense<T, I>(AXcsr.values.data(), AXcsr.graph.row_map.data(),
                            AXcsr.graph.entries.data(), n, m, AX.data());

    csrPartialToDense<T, I>(BXcsr.values.data(), BXcsr.graph.row_map.data(),
                            BXcsr.graph.entries.data(), n, m, BX.data());

    // check();

    kh_AX.destroy_spgemm_handle();
    kh_BX.destroy_spgemm_handle();

    // KokkosBlas::gemm("N", "T", 1.0, A_nm, v, 0.0, AX);  // AX = A * v
    // KokkosBlas::gemm("N", "T", 1.0, B_nm, v, 0.0, BX);  // BX = B * v
  } else {
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);         // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);         // BX = B * X
    Kokkos::deep_copy(tmp, X_AX_BX);                        // tmp = X_AX_BX
    KokkosBlas::gemm("N", "T", 1.0, tmp, v, 0.0, X_AX_BX);  // X = X * v
  }

  // printmat("X", X.data(), 1, m);
  // printmat("AX", AX.data(), 1, m);
  // printmat("BX", BX.data(), 1, m);

  // check();

  View2D<T> M;
  if (Mp != nullptr) {
    M = View2D<T>(Mp, n, n);
  }

  /* Compute residual: R = AX - BX * w */
  View2D<T> R("R = AX - BX * w", n, m);
  View2D<T> R_("R_ = AX + BX * w", n, m);
  compute_residual(AX, BX, w, n, m, R, R_);

  /* Initial: norm for [Xi, Wi, Pi] */
  View2D<T> norm("norm of [Xi, Wi, Pi, R_i]", m, 4);
  compute_norm(X, R, n, m, norm);

  /* Initial for outer loop */
  View2D<T> gramAB_outer("vstack: [gramA_outer, gramB_outer]", 2 * m, m);
  auto gramA_outer = Kokkos::subview(gramAB_outer, Kokkos::make_pair(0, m), Kokkos::ALL());
  auto gramB_outer = Kokkos::subview(gramAB_outer, Kokkos::make_pair(m, 2 * m), Kokkos::ALL());
  View1D<T> w_outer("outer eigenvalues", m);
  View2D<T> v_outer("outer eigenvectors", m, m);

  /* Initial for inner loop */
  Kokkos::View<T[6][3]> gramAB_inner("vstack: [gramA_inner, gramB_inner]");
  auto gramA_inner = Kokkos::subview(gramAB_inner, Kokkos::make_pair(0, 3), Kokkos::ALL());
  auto gramB_inner = Kokkos::subview(gramAB_inner, Kokkos::make_pair(3, 6), Kokkos::ALL());
  View1D<T> w_inner("inner eigenvalues", m);
  View2D<T> v_inner("inner eigenvectors", m, 3);

  /* Initial convergent array as all false: 0 in host */
  View1D<T> is_convergent("convergent flag", m);
  View1D<T> res("residual norm stored in host", m);

  // check();

  /* Start outer loop */
  for (int k = 0; k < maxIter; k++) {
    if (Mp != nullptr) {                              // with preconditioning, normally M = A^-1
      KokkosBlas::gemm("N", "N", 1.0, M, R, 0.0, W);  // W = M * R
    } else {                                          // without preconditioning
      Kokkos::deep_copy(W, R);  // since R not stored in contiguous memory, use deep_copy
    }

    if (k == 1) {
      crsMat_t APcsr;
      crsMat_t BPcsr;

      KernelHandle kh_AP;
      KernelHandle kh_BP;

      kh_AP.create_spgemm_handle(KokkosSparse::StringToSPGEMMAlgorithm(alg));
      kh_BP.create_spgemm_handle(KokkosSparse::StringToSPGEMMAlgorithm(alg));

      Kokkos::parallel_for(
          "copy P to Pcsr.values", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, m}),
          KOKKOS_LAMBDA(int i, int j) { Pcsr.values[i * m + j] = P(i, j); });

      // check();

      KokkosSparse::spgemm_symbolic(kh_AP, Acsr, 0, Pcsr, 0, APcsr);
      KokkosSparse::spgemm_numeric(kh_AP, Acsr, 0, Pcsr, 0, APcsr);

      KokkosSparse::spgemm_symbolic(kh_BP, Bcsr, 0, Pcsr, 0, BPcsr);
      KokkosSparse::spgemm_numeric(kh_BP, Bcsr, 0, Pcsr, 0, BPcsr);

      // check();
      csrPartialToDense<T, I>(APcsr.values.data(), APcsr.graph.row_map.data(),
                              APcsr.graph.entries.data(), n, m, AP.data());
      csrPartialToDense<T, I>(BPcsr.values.data(), BPcsr.graph.row_map.data(),
                              BPcsr.graph.entries.data(), n, m, BP.data());

      // check();

      kh_AP.destroy_spgemm_handle();
      kh_BP.destroy_spgemm_handle();
    }

    crsMat_t AWcsr;
    crsMat_t BWcsr;

    KernelHandle kh_AW;
    KernelHandle kh_BW;

    // kh_AX.set_team_work_size(16);
    // kh_BX.set_team_work_size(16);

    // kh_AX.set_dynamic_scheduling(true);
    // kh_BX.set_dynamic_scheduling(true);

    // check();

    kh_AW.create_spgemm_handle(KokkosSparse::StringToSPGEMMAlgorithm(alg));
    kh_BW.create_spgemm_handle(KokkosSparse::StringToSPGEMMAlgorithm(alg));

    Kokkos::parallel_for(
        "copy W to Wcsr.values", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, m}),
        KOKKOS_LAMBDA(int i, int j) { Wcsr.values[i * m + j] = W(i, j); });

    KokkosSparse::spgemm_symbolic(kh_AW, Acsr, 0, Wcsr, 0, AWcsr);
    KokkosSparse::spgemm_numeric(kh_AW, Acsr, 0, Wcsr, 0, AWcsr);

    KokkosSparse::spgemm_symbolic(kh_BW, Bcsr, 0, Wcsr, 0, BWcsr);
    KokkosSparse::spgemm_numeric(kh_BW, Bcsr, 0, Wcsr, 0, BWcsr);

    // check();
    csrPartialToDense<T, I>(AWcsr.values.data(), AWcsr.graph.row_map.data(),
                            AWcsr.graph.entries.data(), n, m, AW.data());
    csrPartialToDense<T, I>(BWcsr.values.data(), BWcsr.graph.row_map.data(),
                            BWcsr.graph.entries.data(), n, m, BW.data());

    // check();

    kh_AW.destroy_spgemm_handle();
    kh_BW.destroy_spgemm_handle();

    // if (k == 0) {
    //   printmat("W", W, 1, m);
    //   printmat("R", R, 1, m);
    //   printmat("AP", AP, 1, m);
    //   printmat("BP", BP, 1, m);
    //   printmat("AW", AW, 1, m);
    //   printmat("BW", BW, 1, m);
    // }

    /* Perform Rayleigh-Ritz procedure */
    /* Normalize [Xi, Wi, Pi] and store them in contiguous memory for each sub-block */
    Kokkos::parallel_for(
        "vstack: S = [Xi, Wi, Pi], m sub-blocks[n, 3]"
        "vstack: ABS = [AXi, AWi, APi, BXi, BWi, BPi], m sub-blocks[n, 6]",
        n, KOKKOS_LAMBDA(const int i) {
          for (int j = 0; j < m; ++j) {
            if (is_convergent(j) == 0) {
              T X_norm = sqrt(norm(j, 0));
              T W_norm = sqrt(norm(j, 1));
              T P_norm = sqrt(norm(j, 2));

              S(i + n * j, 0) = X(i, j) /= X_norm;
              S(i + n * j, 1) = W(i, j) /= W_norm;
              S(i + n * j, 2) = P(i, j) /= P_norm;

              ABS(i + n * j, 0) = AX(i, j) /= X_norm;
              ABS(i + n * j, 1) = AW(i, j) /= W_norm;
              ABS(i + n * j, 2) = AP(i, j) /= P_norm;
              ABS(i + n * j, 3) = BX(i, j) /= X_norm;
              ABS(i + n * j, 4) = BW(i, j) /= W_norm;
              ABS(i + n * j, 5) = BP(i, j) /= P_norm;
            }
          }
        });

    /* Perform inner Rayleigh-Ritz procedure */
    int n_inner = (k == 0) ? 2 : 3;
    // check();

    /* Use hard lock technique to lock the convergent eigenpairs */
    for (int i = 0; i < m; i++) {
      if (is_convergent(i) == 0) {
        /* Compute symmetric Gram matrices */
        auto w_inner_i = Kokkos::subview(w_inner, i);
        auto v_inner_i = Kokkos::subview(v_inner, i, Kokkos::ALL());

        auto Si = Kokkos::subview(S, Kokkos::make_pair(i * n, (i + 1) * n), Kokkos::ALL());
        auto ABSi = Kokkos::subview(ABS, Kokkos::make_pair(i * n, (i + 1) * n), Kokkos::ALL());
        // check();

        KokkosBlas::gemm("T", "N", 1.0, ABSi, Si, 0.0, gramAB_inner);

        /* Make sure store gramA, gramB to contigous memory */
        if (k == 0) {
          gramA_inner(0, 2) = gramA_inner(1, 0);
          gramA_inner(1, 0) = gramA_inner(1, 1);
          gramB_inner(0, 2) = gramB_inner(1, 0);
          gramB_inner(1, 0) = gramB_inner(1, 1);
        }
        // check();

        // printmat("gramA_inner", gramA_inner.data(), 3, 3);
        // printmat("gramB_inner", gramB_inner.data(), 3, 3);

        // /* Compute eigenvalues and eigenvectors 3x3 eigenvalue problem */
        // sygvx3x3(gramA_inner.data(), gramB_inner.data(), n_inner, 1, w_inner_i.data(),
        //          v_inner_i.data());

        /* Alternative way is to use lapack */
        lapackage::sygvx<T>(gramA_inner.data(), gramB_inner.data(), n_inner, 1, w_inner_i.data(),
                            v_inner_i.data());
        // check();
      }
    }

    // printmat("w_inner", w_inner.data(), m, 1);
    // printmat("v_inner", v_inner.data(), m, 3);

    // check();

    /* Compute the Ritz vector, compute batchly out of inner loop */
    Kokkos::parallel_for(
        "P = W * v(1) + P * v(2), X = X * v(0) + P",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {3 * n, m}),
        KOKKOS_LAMBDA(const int i, const int j) {
          P_AP_BP(i, j) = W_AW_BW(i, j) * v_inner(j, 1) + P_AP_BP(i, j) * v_inner(j, 2);
          X_AX_BX(i, j) = X_AX_BX(i, j) * v_inner(j, 0) + P_AP_BP(i, j);
        });
    // check();

    /* Perform outer Rayleigh-Ritz procedure */
    KokkosBlas::gemm("T", "N", 1.0, X, AX, 0.0, gramA_outer);  // gramA = X^T * AX
    KokkosBlas::gemm("T", "N", 1.0, X, BX, 0.0, gramB_outer);  // gramB = X^T * BX
    // check();

    // printmat("gramA_outer", gramA_outer.data(), m, m);
    // printmat("gramB_outer", gramB_outer.data(), m, m);

    /* Compute eigenvalues and eigenvectors for m x m eigenvalue problem */
    lapackage::sygvx<T>(gramA_outer.data(), gramB_outer.data(), m, m, w_outer.data(),
                        v_outer.data());
    // check();

    /* [X, AX, BX, P, AP, BP] = [X, AX, BX, P, AP, BP] * v */
    Kokkos::deep_copy(tmp, X_AX_BX);
    KokkosBlas::gemm("N", "T", 1.0, tmp, v_outer, 0.0, X_AX_BX);
    Kokkos::deep_copy(tmp, P_AP_BP);
    KokkosBlas::gemm("N", "T", 1.0, tmp, v_outer, 0.0, P_AP_BP);
    // check();

    /* R = AX - BX * w, R_ = AX + BX * w */
    compute_residual(AX, BX, w_outer, n, m, R, R_);
    // check();

    /* Update norm of Xi, Ri, Pi, R_i */
    compute_norm(X, R, P, R_, is_convergent, n, m, m0, norm);
    // check();

    /* Compute residual nrom = ||R|| / ||R_|| = ||AX - BX * w|| / ||AX + BX * w|| */
    compute_residual_norm(norm, is_convergent, m0, res);
    // check();

    /* Check convergence */
    bool converged =
        check_convergence(res.data(), is_convergent.data(), m0, k, maxIter, tol, verbose);
    // check();

    if (converged) break;
  }  // end outer loop

  /* Copy result back to wp, vp */
  View1D<T> w_result(wp, m0);
  View2D<T> v_result(vp, n, m0);
  auto X_m0 = Kokkos::subview(X, Kokkos::ALL(), Kokkos::make_pair(0, m0));
  auto w_m0 = Kokkos::subview(w_outer, Kokkos::make_pair(0, m0));
  Kokkos::deep_copy(v_result, X_m0);
  Kokkos::deep_copy(w_result, w_m0);
}

template <typename T>
void lobpcgI(T* Ap, T* Bp, int n, int m, T* wp, T* vp, T* Xp = nullptr, T* Mp = nullptr,
             double tol = 1e-8, int maxIter = 500, bool verbose = true) {
  // TODO: Bp is nullptr, use identity matrix
  // TODO: sygv for full matrix eigenvalue problem
  // TODO: if not converged, pick up the result for lowest residual
  // TODO: if the lowest residual is still above 1e-4, then fail
  // TODO: print residual precision based on tolerance set
  // TODO: Xp need to copy to device

  const int m0 = m;                    // number of eigenpairs desired
  const int m1 = int(ceil(3.0 * m0));  // added number of eigenpairs to compute
  m = m0 + m1;                         // total number of eigenpairs to compute
  m = n < m ? n : m;                   // m cannot be larger than n

  /* If n < 100, use preconditioned sygvx */
  // if (n < 200) {
  //   m = n;
  // }

  // if (m0 >= int(floor(n * 0.3))) {
  //   printf("\033[1;31mWarning\033[0m: m is larger than 30%% of n.\n");
  //   return;
  // }

  // if (m >= int(floor(n * 0.3))) {
  //   m = int(floor(n * 0.3)) - 1;
  // }

  if (verbose) {
    printf(
        "lobpcg: n = \033[32m%d\033[0m, m = \033[32m%d\033[0m, m (added) = "
        "\033[32m%d\033[0m\n",
        n, m0, m - m0);
  }

  View2D<T> A(Ap, n, n);
  View2D<T> B(Bp, n, n);

  /* store in vstack [ X | AX | BX ], [ P | AP | BP ], [ W | AW | BW ] */
  View2D<T> X_AX_BX("vstack: [ X | AX | BX ]", 3 * n, m);
  View2D<T> W_AW_BW("vstack: [ W | AW | BW ]", 3 * n, m);
  View2D<T> P_AP_BP("vstack: [ P | AP | BP ]", 3 * n, m);

  View2D<T> X = Kokkos::subview(X_AX_BX, Kokkos::make_pair(0, n), Kokkos::ALL());
  View2D<T> W = Kokkos::subview(W_AW_BW, Kokkos::make_pair(0, n), Kokkos::ALL());
  View2D<T> P = Kokkos::subview(P_AP_BP, Kokkos::make_pair(0, n), Kokkos::ALL());

  View2D<T> AX = Kokkos::subview(X_AX_BX, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());
  View2D<T> AW = Kokkos::subview(W_AW_BW, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());
  View2D<T> AP = Kokkos::subview(P_AP_BP, Kokkos::make_pair(n, 2 * n), Kokkos::ALL());

  View2D<T> BX = Kokkos::subview(X_AX_BX, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());
  View2D<T> BW = Kokkos::subview(W_AW_BW, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());
  View2D<T> BP = Kokkos::subview(P_AP_BP, Kokkos::make_pair(2 * n, 3 * n), Kokkos::ALL());

  View2D<T> S("vstack: [Xi, Wi, Pi], m sub-blocks[n, 3]", m * n, 3);
  View2D<T> ABS("vstack: [AXi, AWi, APi, BXi, BWi, BPi], m sub-blocks[n, 6]", m * n, 6);

  View2D<T> tmp("temp for X_AX_BX", 3 * n, m);

  /* Compute XAX0 = X.T * A * X, XBX0 = X.T * B * X */
  View2D<T> XAX0("XAX0", m, m);
  View2D<T> XBX0("XBX0", m, m);

  if (Xp == nullptr) {
    Kokkos::parallel_for(
        "lobpcg::set: XAX0, XBX0", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m, m}),
        KOKKOS_LAMBDA(int i, int j) {
          XAX0(i, j) = A(i, j);  // X = eye(n, m) -> XAX0 = A[:m, :m]
          XBX0(i, j) = B(i, j);  // X = eye(n, m) -> XBX0 = B[:m, :m]
        });
  } else {
    X = View2D<T>(Xp, n, m);
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);     // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);     // BX = B * X
    KokkosBlas::gemm("T", "N", 1.0, X, AX, 0.0, XAX0);  // XAX0 = X.T * AX
    KokkosBlas::gemm("T", "N", 1.0, X, BX, 0.0, XBX0);  // XBX0 = X.T * BX
  }

  /* Solve generalized eigenvalue problem to get initial eigenvectors */
  View1D<T> w("eigenvalues", m);
  View2D<T> v("eigenvectors column major", m, m);

  lapackage::sygvx<T>(XAX0.data(), XBX0.data(), m, m, w.data(), v.data());

  /* Compute: X = X * v, AX = A * X * v, BX = B * X * v */
  if (Xp == nullptr) {
    auto A_nm = Kokkos::subview(A, Kokkos::ALL(), Kokkos::make_pair(0, m));
    auto B_nm = Kokkos::subview(B, Kokkos::ALL(), Kokkos::make_pair(0, m));

    Kokkos::parallel_for(
        "X = eye(n, m) -> X = hstack(v, 0)", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m, m}),
        KOKKOS_LAMBDA(int i, int j) { X(i, j) = v(j, i); });

    KokkosBlas::gemm("N", "T", 1.0, A_nm, v, 0.0, AX);  // AX = A * v
    KokkosBlas::gemm("N", "T", 1.0, B_nm, v, 0.0, BX);  // BX = B * v
  } else {
    View2D<T> X0 = X;
    KokkosBlas::gemm("N", "N", 1.0, A, X, 0.0, AX);         // AX = A * X
    KokkosBlas::gemm("N", "N", 1.0, B, X, 0.0, BX);         // BX = B * X
    Kokkos::deep_copy(tmp, X_AX_BX);                        // tmp = X_AX_BX
    KokkosBlas::gemm("N", "T", 1.0, tmp, v, 0.0, X_AX_BX);  // X = X * v
  }

  View2D<T> M;
  if (Mp != nullptr) {
    M = View2D<T>(Mp, n, n);
  }

  /* Compute residual: R = AX - BX * w */
  View2D<T> R("R = AX - BX * w", n, m);
  View2D<T> R_("R_ = AX + BX * w", n, m);
  compute_residual(AX, BX, w, n, m, R, R_);

  /* Initial: norm for [Xi, Wi, Pi] */
  View2D<T> norm("norm of [Xi, Wi, Pi, R_i]", m, 4);
  compute_norm(X, R, n, m, norm);

  /* Initial for outer loop */
  View2D<T> gramAB_outer("vstack: [gramA_outer, gramB_outer]", 2 * m, m);
  auto gramA_outer = Kokkos::subview(gramAB_outer, Kokkos::make_pair(0, m), Kokkos::ALL());
  auto gramB_outer = Kokkos::subview(gramAB_outer, Kokkos::make_pair(m, 2 * m), Kokkos::ALL());
  View1D<T> w_outer("outer eigenvalues", m);
  View2D<T> v_outer("outer eigenvectors", m, m);

  /* Initial for inner loop */
  Kokkos::View<T[6][3]> gramAB_inner("vstack: [gramA_inner, gramB_inner]");
  auto gramA_inner = Kokkos::subview(gramAB_inner, Kokkos::make_pair(0, 3), Kokkos::ALL());
  auto gramB_inner = Kokkos::subview(gramAB_inner, Kokkos::make_pair(3, 6), Kokkos::ALL());
  View1D<T> w_inner("inner eigenvalues", m);
  View2D<T> v_inner("inner eigenvectors", m, 3);

  /* Initial convergent array as all false: 0 in host */
  View1D<T> is_convergent("convergent flag", m);
  View1D<T> res("residual norm stored in host", m);

  /* Start outer loop */
  for (int k = 0; k < maxIter; k++) {
    if (Mp != nullptr) {                              // with preconditioning, normally M = A^-1
      KokkosBlas::gemm("N", "N", 1.0, M, R, 0.0, W);  // W = M * R
    } else {                                          // without preconditioning
      Kokkos::deep_copy(W, R);  // since R not stored in contiguous memory, use deep_copy
    }

    if (k == 1) {
      KokkosBlas::gemm("N", "N", 1.0, A, P, 0.0, AP);
      KokkosBlas::gemm("N", "N", 1.0, B, P, 0.0, BP);
    }

    KokkosBlas::gemm("N", "N", 1.0, A, W, 0.0, AW);
    KokkosBlas::gemm("N", "N", 1.0, B, W, 0.0, BW);

    /* Perform Rayleigh-Ritz procedure */
    /* Normalize [Xi, Wi, Pi] and store them in contiguous memory for each sub-block */
    Kokkos::parallel_for(
        "vstack: S = [Xi, Wi, Pi], m sub-blocks[n, 3]"
        "vstack: ABS = [AXi, AWi, APi, BXi, BWi, BPi], m sub-blocks[n, 6]",
        n, KOKKOS_LAMBDA(const int i) {
          for (int j = 0; j < m; ++j) {
            if (is_convergent(j) == 0) {
              T X_norm = sqrt(norm(j, 0));
              T W_norm = sqrt(norm(j, 1));
              T P_norm = sqrt(norm(j, 2));

              S(i + n * j, 0) = X(i, j) /= X_norm;
              S(i + n * j, 1) = W(i, j) /= W_norm;
              S(i + n * j, 2) = P(i, j) /= P_norm;

              ABS(i + n * j, 0) = AX(i, j) /= X_norm;
              ABS(i + n * j, 1) = AW(i, j) /= W_norm;
              ABS(i + n * j, 2) = AP(i, j) /= P_norm;
              ABS(i + n * j, 3) = BX(i, j) /= X_norm;
              ABS(i + n * j, 4) = BW(i, j) /= W_norm;
              ABS(i + n * j, 5) = BP(i, j) /= P_norm;
            }
          }
        });

    /* Perform inner Rayleigh-Ritz procedure */
    int n_inner = (k == 0) ? 2 : 3;

    /* Use hard lock technique to lock the convergent eigenpairs */
    for (int i = 0; i < m; i++) {
      if (is_convergent(i) == 0) {
        /* Compute symmetric Gram matrices */
        auto w_inner_i = Kokkos::subview(w_inner, i);
        auto v_inner_i = Kokkos::subview(v_inner, i, Kokkos::ALL());

        auto Si = Kokkos::subview(S, Kokkos::make_pair(i * n, (i + 1) * n), Kokkos::ALL());
        auto ABSi = Kokkos::subview(ABS, Kokkos::make_pair(i * n, (i + 1) * n), Kokkos::ALL());

        KokkosBlas::gemm("T", "N", 1.0, ABSi, Si, 0.0, gramAB_inner);

        /* Make sure store gramA, gramB to contigous memory */
        if (k == 0) {
          gramA_inner(0, 2) = gramA_inner(1, 0);
          gramA_inner(1, 0) = gramA_inner(1, 1);
          gramB_inner(0, 2) = gramB_inner(1, 0);
          gramB_inner(1, 0) = gramB_inner(1, 1);
        }

        // /* Compute eigenvalues and eigenvectors 3x3 eigenvalue problem */
        // sygvx3x3(gramA_inner.data(), gramB_inner.data(), n_inner, 1, w_inner_i.data(),
        //          v_inner_i.data());

        /* Alternative way is to use lapack */
        lapackage::sygvx<T>(gramA_inner.data(), gramB_inner.data(), n_inner, 1, w_inner_i.data(),
                            v_inner_i.data());
      }
    }

    // printmat("w_inner", w_inner.data(), m, 1);
    // printmat("v_inner", v_inner.data(), m, 3);

    /* Compute the Ritz vector, compute batchly out of inner loop */
    Kokkos::parallel_for(
        "P = W * v(1) + P * v(2), X = X * v(0) + P",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {3 * n, m}),
        KOKKOS_LAMBDA(const int i, const int j) {
          P_AP_BP(i, j) = W_AW_BW(i, j) * v_inner(j, 1) + P_AP_BP(i, j) * v_inner(j, 2);
          X_AX_BX(i, j) = X_AX_BX(i, j) * v_inner(j, 0) + P_AP_BP(i, j);
        });

    /* Perform outer Rayleigh-Ritz procedure */
    KokkosBlas::gemm("T", "N", 1.0, X, AX, 0.0, gramA_outer);  // gramA = X^T * AX
    KokkosBlas::gemm("T", "N", 1.0, X, BX, 0.0, gramB_outer);  // gramB = X^T * BX

    /* Compute eigenvalues and eigenvectors for m x m eigenvalue problem */
    lapackage::sygvx<T>(gramA_outer.data(), gramB_outer.data(), m, m, w_outer.data(),
                        v_outer.data());

    /* [X, AX, BX, P, AP, BP] = [X, AX, BX, P, AP, BP] * v */
    Kokkos::deep_copy(tmp, X_AX_BX);
    KokkosBlas::gemm("N", "T", 1.0, tmp, v_outer, 0.0, X_AX_BX);
    Kokkos::deep_copy(tmp, P_AP_BP);
    KokkosBlas::gemm("N", "T", 1.0, tmp, v_outer, 0.0, P_AP_BP);
    /* R = AX - BX * w, R_ = AX + BX * w */
    compute_residual(AX, BX, w_outer, n, m, R, R_);

    /* Update norm of Xi, Ri, Pi, R_i */
    compute_norm(X, R, P, R_, is_convergent, n, m, m0, norm);

    /* Compute residual nrom = ||R|| / ||R_|| = ||AX - BX * w|| / ||AX + BX * w|| */
    compute_residual_norm(norm, is_convergent, m0, res);

    /* Check convergence */
    bool converged =
        check_convergence(res.data(), is_convergent.data(), m0, k, maxIter, tol, verbose);

    if (converged) break;
  }  // end outer loop

  /* Copy result back to wp, vp */
  View1D<T> w_result(wp, m0);
  View2D<T> v_result(vp, n, m0);
  auto X_m0 = Kokkos::subview(X, Kokkos::ALL(), Kokkos::make_pair(0, m0));
  auto w_m0 = Kokkos::subview(w_outer, Kokkos::make_pair(0, m0));
  Kokkos::deep_copy(v_result, X_m0);
  Kokkos::deep_copy(w_result, w_m0);
}

template <typename T, typename I>
void lobpcg(T Ax[], I Ap[], I Aj[], T Bx[], I Bp[], I Bj[], int n, int m, T wp[], T vp[], T Mp[],
            T Xp[] = nullptr, double tol = 1e-8, int maxIter = 500, bool verbose = true) {
#ifdef KOKKOS_ENABLE_CUDA
  printf("\033[1;31mWarning\033[0m: lobpcg_sparse is not supported on GPU yet.\n");
#else
  T* A = new T[n * n];
  T* B = new T[n * n];
  csrPartialToDense<T, I>(Ax, Ap, Aj, n, n, A);
  csrPartialToDense<T, I>(Bx, Bp, Bj, n, n, B);

  // tick("lobpcg");
  // lobpcgI(A, B, n, m, wp, vp, Xp, Mp, tol, maxIter, verbose);
  // tock("lobpcg");

  delete[] A;
  delete[] B;

  tick("lobpcgII");
  lobpcgII(Ax, Ap, Aj, Bx, Bp, Bj, n, m, wp, vp, Xp, Mp, tol, maxIter, verbose);
  tock("lobpcgII");
#endif
}

}  // namespace linalg::sparse

#endif  // LOBPCG_SPARSE_HPP_