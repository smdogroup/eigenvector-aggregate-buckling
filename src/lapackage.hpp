#pragma once
#ifndef LAPACKAGE_HPP
#define LAPACKAGE_HPP

#include <lapacke.h>

namespace lapackage {
/********************* declarations *********************/
/**
 * Computes eigenvalues and eigenvectors for Ax=¦ËBx using LAPACK.
 *
 * Input:
 *    A: symmetric matrix
 *    B: symmetric positive definite matrix
 *    n: dimension of A and B
 *    m: number of eigenvalues and eigenvectors to be computed
 *
 * Output:
 *    eigenvalues: eigenvalues of Ax=¦ËBx
 *    eigenvectors: eigenvectors of Ax=¦ËBx
 *
 * Note:
 *    eigenvalues and eigenvectors are sorted in ascending order
 */
template <typename T>
void sygvx(T* A, T* B, int n, int m, T* eigenvalues, T* eigenvectors, double tol);

/**
 * Computes the inverse of a matrix using LU factorization with partial
 * pivoting and row interchanges.
 *
 * Input:
 *    A: matrix to be inverted
 *    n: dimension of A
 *
 * Output:
 *    A: inverse of A
 */
template <typename T>
void inverse(T* A, int n, T* Ainv);

/******************** LAPACK routines *******************/
void lapack_getrf(int* m, int* n, double* A, int* lda, int* ipiv, int* info) {
  return LAPACK_dgetrf(m, n, A, lda, ipiv, info);
}

void lapack_getri(int* n, double* A, int* lda, int* ipiv, double* work,
                  int* lwork, int* info) {
  return LAPACK_dgetri(n, A, lda, ipiv, work, lwork, info);
}

void lapack_sygvx(int* itype, char* jobz, char* range, char* uplo, int* n,
                  double* a, int* lda, double* b, int* ldb, double* vl,
                  double* vu, int* il, int* iu, double* abstol, int* m,
                  double* w, double* z, int* ldz, double* work, int* lwork,
                  int* iwork, int* ifail, int* info) {
  return LAPACK_dsygvx(itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il,
                       iu, abstol, m, w, z, ldz, work, lwork, iwork, ifail,
                       info);
}

/********************** definitions *********************/
/* implementation of inverse() */
template <typename T>
void inverse(T* A, int n, T* Ainv) {
    for (int i = 0; i < n * n; i++) {
    Ainv[i] = A[i];
  }

  int* ipiv = new int[n];
  int lwork = n * n;
  double* work = new double[lwork];
  int info;

  lapack_getrf(&n, &n, Ainv, &n, ipiv, &info);
  lapack_getri(&n, Ainv, &n, ipiv, work, &lwork, &info);

  delete[] ipiv;
  delete[] work;
}

/* implementation of sygvx() */
template <typename T>
void sygvx(double* A, double* B, int n, int m, double* eigenvalues,
           double* eigenvectors, double tol=0.0) {
  int itype = 1;        // Ax = ¦ËBx
  char jobz = 'V';      // Compute eigenvalues and eigenvectors
  char range = 'I';     // Compute in il-th through iu-th range
  char uplo = 'U';      // Upper triangular part of A and B are stored
  int lda = n;          // Leading dimension of A
  int ldb = n;          // Leading dimension of B
  T vl = 0;             // Lower bound of eigenvalues not referenced
  T vu = 0;             // Upper bound of eigenvalues not referenced
  int il = 1;           // Index of smallest eigenvalue to compute
  int iu = m;           // Index of largest eigenvalue to compute
  double abstol = tol;  // Absolute tolerance

  // Initialize work arrays
  int lwork = 10 * n;  // Work space size

  // Work space query
  double* work = new double[lwork];
  int* iwork = new int[5 * n];
  int* ifail = new int[n];
  int info;

  // Compute eigenvalues and eigenvectors
  lapack_sygvx(&itype, &jobz, &range, &uplo, &n, A, &lda, B, &ldb, &vl, &vu,
               &il, &iu, &abstol, &m, eigenvalues, eigenvectors, &n, work,
               &lwork, iwork, ifail, &info);

  // Check for errors
  if (info != 0) {
    throw std::runtime_error("Error: sygvx returned " + std::to_string(info));
  }

  delete[] work;
  delete[] iwork;
  delete[] ifail;
}

}  // namespace lapackage

#endif  // LAPACKAGE_HPP