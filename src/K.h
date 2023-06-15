#ifndef STIFFNESS_H
#define STIFFNESS_H

#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_mult.hpp>
#include <KokkosBlas1_reciprocal.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas1_update.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <Kokkos_Core.hpp>
#include <cstring>

#include "converter.h"
#include "toolkit.h"

// #include <KokkosSparse_gmres.hpp>
// #include <Kokkos_Sort.hpp>
// #include <tuple>
// #include <vector>

// #include "KokkosKernels_default_types.hpp"
// #include "KokkosSparse_CrsMatrix.hpp"
// #include "KokkosSparse_IOUtils.hpp"
// #include "KokkosSparse_spmv.hpp"


// struct CompareSparseEntry {
//   KOKKOS_INLINE_FUNCTION
//   bool operator()(const std::tuple<int, int, double>& entry1,
//                   const std::tuple<int, int, double>& entry2) const {
//     return std::get<0>(entry1) < std::get<0>(entry2) ||
//            (std::get<0>(entry1) == std::get<0>(entry2) &&
//             std::get<1>(entry1) < std::get<1>(entry2));
//   }
// };

// template <typename crsMat_t>
// void makeSparseMatrix(
//     typename crsMat_t::StaticCrsGraphType::row_map_type::non_const_type& ptr,
//     typename crsMat_t::StaticCrsGraphType::entries_type::non_const_type& ind,
//     typename crsMat_t::values_type::non_const_type& val,
//     typename crsMat_t::ordinal_type& numRows,
//     typename crsMat_t::ordinal_type& numCols, typename crsMat_t::size_type& nnz,
//     const int whichMatrix) {
//   typedef typename crsMat_t::StaticCrsGraphType::row_map_type::non_const_type
//       ptr_type;
//   typedef typename crsMat_t::StaticCrsGraphType::entries_type::non_const_type
//       ind_type;
//   typedef typename crsMat_t::values_type::non_const_type val_type;
//   typedef typename crsMat_t::ordinal_type lno_t;
//   typedef typename crsMat_t::size_type size_type;
//   typedef typename crsMat_t::value_type scalar_t;

//   using Kokkos::HostSpace;
//   using Kokkos::MemoryUnmanaged;
//   using Kokkos::View;

//   if (whichMatrix == 0) {
//     numCols = numRows;
//     nnz = 2 + 3 * (numRows - 2) + 2;
//     size_type* ptrRaw = new size_type[numRows + 1];
//     lno_t* indRaw = new lno_t[nnz];
//     scalar_t* valRaw = new scalar_t[nnz];
//     scalar_t two = 2.0;
//     scalar_t mone = -1.0;

//     // Add rows one-at-a-time
//     for (int i = 0; i < (numRows + 1); i++) {
//       if (i == 0) {
//         ptrRaw[0] = 0;
//         indRaw[0] = 0;
//         indRaw[1] = 1;
//         valRaw[0] = two;
//         valRaw[1] = mone;
//       } else if (i == numRows) {
//         ptrRaw[numRows] = nnz;
//       } else if (i == (numRows - 1)) {
//         ptrRaw[i] = 2 + 3 * (i - 1);
//         indRaw[2 + 3 * (i - 1)] = i - 1;
//         indRaw[2 + 3 * (i - 1) + 1] = i;
//         valRaw[2 + 3 * (i - 1)] = mone;
//         valRaw[2 + 3 * (i - 1) + 1] = two;
//       } else {
//         ptrRaw[i] = 2 + 3 * (i - 1);
//         indRaw[2 + 3 * (i - 1)] = i - 1;
//         indRaw[2 + 3 * (i - 1) + 1] = i;
//         indRaw[2 + 3 * (i - 1) + 2] = i + 1;
//         valRaw[2 + 3 * (i - 1)] = mone;
//         valRaw[2 + 3 * (i - 1) + 1] = two;
//         valRaw[2 + 3 * (i - 1) + 2] = mone;
//       }
//     }

//     // Create the output Views.
//     ptr = ptr_type("ptr", numRows + 1);
//     ind = ind_type("ind", nnz);
//     val = val_type("val", nnz);

//     // Wrap the above three arrays in unmanaged Views, so we can use deep_copy.
//     typename ptr_type::HostMirror::const_type ptrIn(ptrRaw, numRows + 1);
//     typename ind_type::HostMirror::const_type indIn(indRaw, nnz);
//     typename val_type::HostMirror::const_type valIn(valRaw, nnz);

//     Kokkos::deep_copy(ptr, ptrIn);
//     Kokkos::deep_copy(ind, indIn);
//     Kokkos::deep_copy(val, valIn);

//     delete[] ptrRaw;
//     delete[] indRaw;
//     delete[] valRaw;
//   } else {  // whichMatrix != 0
//     std::ostringstream os;
//     os << "Invalid whichMatrix value " << whichMatrix
//        << ".  Valid value(s) include " << 0 << ".";
//     throw std::invalid_argument(os.str());
//   }
// }

// template <typename crsMat_t>
// crsMat_t makeCrsMatrix(int numRows) {
//   typedef typename crsMat_t::StaticCrsGraphType graph_t;
//   typedef typename graph_t::row_map_type::non_const_type lno_view_t;
//   typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
//   typedef typename crsMat_t::values_type::non_const_type scalar_view_t;
//   typedef typename crsMat_t::ordinal_type lno_t;
//   typedef typename crsMat_t::size_type size_type;

//   lno_view_t ptr;
//   lno_nnz_view_t ind;
//   scalar_view_t val;
//   lno_t numCols;
//   size_type nnz;

//   const int whichMatrix = 0;
//   makeSparseMatrix<crsMat_t>(ptr, ind, val, numRows, numCols, nnz, whichMatrix);
//   return crsMat_t("A", numRows, numCols, nnz, val, ptr, ind);
// }


template <typename T>
View1D<T> populateBe(T xi, T eta, const View2D<T>& xe, const View2D<T>& ye,
                View3D<T>& Be) {
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

  return detJ;
}

template <typename T>
View3D<T> computeK(const View2D<T>& X, const View2D<int>& conn,
                   const View1D<T>& rho, const View2D<T>& C0,
                   const View1D<int>& reduced, const T rho0_K,
                   const char* ptype_K, const double p, const double q,
                   View1D<int>& indptr, View1D<int>& indices, View1D<T>& data,
                   View1D<T>& f) {
  const int nelems = conn.extent(0);
  const int nnodes = X.extent(0);
  const int nr = reduced.extent(0);

  View3D<T> C("C", nelems, 3, 3);
  View2D<T> xe("xe", nelems, 4);
  View2D<T> ye("ye", nelems, 4);
  View3D<T> Be("Be", nelems, 3, 8);
  View3D<T> Ke("Ke", nelems, 8, 8);

  // Compute Gauss quadrature with a 2-point quadrature rule
  const double gauss_pts[2] = {-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)};

  // Prepare the shape functions
  Kokkos::parallel_for(
      nelems, KOKKOS_LAMBDA(const int n) {
        // Average the density to get the element - wise density
        T rhoE_n = 0.25 * (rho(conn(n, 0)) + rho(conn(n, 1)) + rho(conn(n, 2)) +
                           rho(conn(n, 3)));

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

        // Get the element-wise solution variables
        // Compute the x and y coordinates of each element
        for (int i = 0; i < 4; ++i) {
          // xe = X[self.conn, 0], ye = X[self.conn, 1]
          xe(n, i) = X(conn(n, i), 0);
          ye(n, i) = X(conn(n, i), 1);
        }
      });

  for (int jj = 0; jj < 2; jj++) {
    for (int ii = 0; ii < 2; ii++) {
      T xi = gauss_pts[ii];
      T eta = gauss_pts[jj];

      auto detJ = populateBe(xi, eta, xe, ye, Be);

      // Ke += np.einsum("n,nij,nik,nkl -> njl", detJ, Be, C, Be)
      Kokkos::parallel_for(
          nelems, KOKKOS_LAMBDA(const int n) {
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

  // Kokkos::Timer timer;

  // // Set up the i-j indices for the matrix - these are the row and column
  // // indices in the stiffness matrix
  // // self.var = np.zeros((self.conn.shape[0], 8), dtype=int)
  // // self.var[:, ::2] = 2 * self.conn self.var[:, 1::2] = 2 * self.conn + 1

  // View2D<int> var("var", nelems, 8);
  // Kokkos::parallel_for(
  //     nelems, KOKKOS_LAMBDA(const int n) {
  //       for (int i = 0; i < 4; i++) {
  //         var(n, i * 2) = 2 * conn(n, i);
  //         var(n, i * 2 + 1) = 2 * conn(n, i) + 1;
  //       }
  //     });

  // View1D<int> i_index("i", nelems * 64);
  // View1D<int> j_index("j", nelems * 64);
  // Kokkos::parallel_for(
  //     nelems, KOKKOS_LAMBDA(const int n) {
  //       for (int ii = 0; ii < 8; ii++) {
  //         for (int jj = 0; jj < 8; jj++) {
  //           i_index(n * 64 + ii * 8 + jj) = var(n, ii);
  //           j_index(n * 64 + ii * 8 + jj) = var(n, jj);
  //         }
  //       }
  //     });

  // // Ke.flatten()
  // View1D<T> Ke_flat("Ke_flat", nelems * 64);
  // Kokkos::parallel_for(
  //     nelems, KOKKOS_LAMBDA(const int n) {
  //       for (int ii = 0; ii < 8; ii++) {
  //         for (int jj = 0; jj < 8; jj++) {
  //           Ke_flat(n * 64 + ii * 8 + jj) = Ke(n, ii, jj);
  //         }
  //       }
  //     });

  // int num_rows = 2 * nnodes;
  // int num_cols = 2 * nnodes;
  // int num_entries = nelems * 64;

  // // Create a Kokkos view of tuples (i_index, j_index, Ke_flat) to represent the
  // // sparse entries
  // Kokkos::View<std::tuple<int, int, double>*> sparse_entries("sparse_entries",
  //                                                            num_entries);
  // Kokkos::parallel_for(
  //     Kokkos::RangePolicy<>(0, num_entries), KOKKOS_LAMBDA(const int i) {
  //       sparse_entries(i) = std::make_tuple(i_index(i), j_index(i), Ke_flat(i));
  //     });
  // Kokkos::fence();

  // // Sort the sparse entries using a custom sorting algorithm
  // // std::sort(sparse_entries.data(), sparse_entries.data() + num_entries,
  // //           CompareSparseEntry());

  // // Remove duplicate entries within each row
  // View1D<int> dup_indices("dup_indices", num_entries);
  // Kokkos::View<int*> new_i_index("new_i_index", num_entries);
  // Kokkos::View<int*> new_j_index("new_j_index", num_entries);
  // Kokkos::View<double*> new_Ke_flat("new_Ke_flat", num_entries);
  // // int num_dup_entries = 0;
  // int prev_row_index = -1;
  // int prev_col_index = -1;
  // int new_num_entries = 0;

  // // remove duplicate entries
  // for (int i = 0; i < num_entries; i++) {
  //   int row_index = std::get<0>(sparse_entries(i));
  //   int col_index = std::get<1>(sparse_entries(i));
  //   T val = std::get<2>(sparse_entries(i));

  //   if (row_index == prev_row_index && col_index == prev_col_index) {
  //     dup_indices(i) = 1;
  //   } else {
  //     dup_indices(i) = 0;
  //     new_i_index(new_num_entries) = row_index;
  //     new_j_index(new_num_entries) = col_index;
  //     new_Ke_flat(new_num_entries) = val;
  //     new_num_entries++;
  //   }
  //   prev_row_index = row_index;
  //   prev_col_index = col_index;
  // }
  // // resize the views
  // Kokkos::resize(new_i_index, new_num_entries);
  // Kokkos::resize(new_j_index, new_num_entries);
  // Kokkos::resize(new_Ke_flat, new_num_entries);

  // Kokkos::View<int*> reduced_i_index("reduced_i_index", nr);
  // Kokkos::View<int*> reduced_j_index("reduced_j_index", nr);
  // Kokkos::View<double*> reduced_Ke_flat("reduced_Ke_flat", nr);

  // // // reduce matrix by view reduced for row and col
  // for (int i = 0; i < new_num_entries; i++) {
  //   int row_index = new_i_index(i);
  //   int col_index = new_j_index(i);
  //   T val = new_Ke_flat(i);

  //   for (int j = 0; j < nr; j++) {
  //     T index = reduced(j);
  //   }
  // }

  // // update the number of rows and columns
  // num_rows = new_i_index(new_num_entries - 1) + 1;
  // num_cols = num_rows;

  // // // convert the coo row indices to csr row pointers, starting at 0
  // View1D<int> row_pointers("row_pointers", num_rows + 1);
  // Kokkos::parallel_for(
  //     num_rows + 1, KOKKOS_LAMBDA(const int i) { row_pointers(i) = 0; });
  // // sorting the i_index array, meanwhile reordering the j_index and Ke_flat

  // Kokkos::parallel_for(
  //     new_num_entries,
  //     KOKKOS_LAMBDA(const int i) { row_pointers(new_i_index(i) + 1)++; });
  // Kokkos::parallel_scan(
  //     num_rows + 1, KOKKOS_LAMBDA(const int i, int& update, const bool final) {
  //       update += row_pointers(i);
  //       if (final) {
  //         row_pointers(i) = update;
  //       }
  //     });

  // using crsMat_t =
  //     KokkosSparse::CrsMatrix<T, int, Kokkos::DefaultExecutionSpace, void, int>;

  // typedef typename crsMat_t::StaticCrsGraphType graph_t;
  // typedef typename graph_t::row_map_type::non_const_type row_map_t;
  // typedef typename graph_t::entries_type::non_const_type entries_t;

  // // create the graph
  // row_map_t row_map("row_map", num_rows + 1);
  // entries_t entries("entries", new_num_entries);

  // Kokkos::parallel_for(
  //     num_rows + 1,
  //     KOKKOS_LAMBDA(const int i) { row_map(i) = row_pointers(i); });
  // Kokkos::parallel_for(
  //     new_num_entries,
  //     KOKKOS_LAMBDA(const int i) { entries(i) = new_j_index(i); });

  // graph_t graph(entries, row_map);

  // checks();

  // // create the values
  // View1D<T> values("values", new_num_entries);
  // Kokkos::parallel_for(
  //     new_num_entries,
  //     KOKKOS_LAMBDA(const int i) { values(i) = new_Ke_flat(i); });

  // // create the crs matrix
  // crsMat_t K("K", num_cols, values, graph);

  // // Typedefs
  // typedef double scalar_type;
  // typedef int ordinal_type;
  // typedef int size_type;
  // typedef Kokkos::DefaultExecutionSpace device_type;
  // typedef KokkosSparse::CrsMatrix<scalar_type, ordinal_type, device_type, void,
  //                                 size_type>
  //     crs_matrix_type;

  // int N = num_rows;
  // // Allocate matrix A on device
  // // crs_matrix_type A = makeCrsMatrix<crs_matrix_type>(2 * nnodes);
  // crs_matrix_type A("A", nr, nr, data.extent(0), data, indptr, indices);

  // // // print size of A
  // // std::cout << "A size: " << A.values.extent(0) << std::endl;

  // // // print number of rows, columns, and entries
  // // std::cout << "num_rows: " << A.numRows() << std::endl;
  // // std::cout << "num_cols: " << A.numCols() << std::endl;
  // // std::cout << "num_entries: " << A.nnz() << std::endl;

  // // // print row_map
  // // std::cout << "row_map: " << std::endl;
  // // for (int i = 0; i < A.graph.row_map.extent(0); i++) {
  // //   std::cout << A.graph.row_map(i) << " ";
  // // }
  // // std::cout << std::endl;

  // // // print entries
  // // std::cout << "entries: " << std::endl;
  // // for (int i = 0; i < A.graph.entries.extent(0); i++) {
  // //   std::cout << A.graph.entries(i) << " ";
  // // }
  // // std::cout << std::endl;

  // // // print values
  // // std::cout << "values: " << std::endl;
  // // for (int i = 0; i < values.extent(0); i++) {
  // //   std::cout << values(i) << " ";
  // // }
  // // std::cout << std::endl;

  // // print values new line each 4 entries
  // // std::cout << "values: " << std::endl;
  // // for (int i = 0; i < A.values.extent(0); i++) {
  // //   if (i % 4 == 0) {
  // //     std::cout << std::endl;
  // //   }
  // //   std::cout << A.values(i) << " ";
  // // }

  // // crs_matrix_type A("A", N, N, data.extent(0), data, indptr, indices);

  // // // Print b
  // // std::cout << "b: " << std::endl;
  // // for (int i = 0; i < 100; i++) {
  // //   std::cout << b(i) << " ";
  // // }
  // // std::cout << std::endl;

  // // std::cout << "row_map: " << std::endl;
  // // for (int i = 0; i < A.graph.row_map.extent(0); i++) {
  // //   std::cout << A.graph.row_map(i) << " ";
  // // }
  // // std::cout << std::endl;

  // // std::cout << "entries: " << std::endl;
  // // for (int i = 0; i < 100; i++) {
  // //   std::cout << A.graph.entries(i) << " ";
  // // }
  // // std::cout << std::endl;

  // // std::cout << "values: " << std::endl;
  // // for (int i = 0; i < 100; i++) {
  // //   for (int j = 0; j < 4; j++) {
  // //     std::cout << A.values(i) << " ";
  // //   }
  // //   std::cout << std::endl;
  // // }
  // // std::cout << std::endl;

  // // //////////////////////

  // using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
  //     typename crsMat_t::const_ordinal_type, typename crsMat_t::const_size_type,
  //     typename crsMat_t::const_value_type,
  //     typename crsMat_t::device_type::execution_space,
  //     typename crsMat_t::device_type::memory_space,
  //     typename crsMat_t::device_type::memory_space>;

  // std::string ortho("CGS2");  // orthog type
  // int m = 50;                 // Max subspace size before restarting.
  // double convTol = 1e-8;      // Relative residual convergence tolerance.
  // int cycLim = 50;            // Maximum number of times to restart the solver.

  // KernelHandle kh;
  // kh.create_gmres_handle(m, convTol, cycLim);
  // auto gmres_handle = kh.get_gmres_handle();
  // // Get full gmres handle type using decltype. Deferencing a pointer gives a
  // // reference, so we need to strip that too.
  // using GMRESHandle =
  //     typename std::remove_reference<decltype(*gmres_handle)>::type;
  // gmres_handle->set_ortho(ortho == "CGS2" ? GMRESHandle::Ortho::CGS2
  //                                         : GMRESHandle::Ortho::MGS);

  // // // set b vector with all equal to 1
  // View1D<T> b("b", nr);
  // Kokkos::parallel_for(
  //     N, KOKKOS_LAMBDA(const int i) { b(i) = 1.0; });

  // // solve the system
  // View1D<T> x("x", nr);

  // double prepare_time = timer.seconds();
  // timer.reset();

  // // use GMRES solver
  // KokkosSparse::Experimental::gmres(&kh, A, f, x);

  // // // print x
  // // std::cout << "x: " << std::endl;
  // // for (int i = 0; i < 100; i++) {
  // //   std::cout << x(i) << " ";
  // // }
  // // std::cout << std::endl;

  // double solve_time = timer.seconds();
  // timer.reset();

  // int printInfo = 1;

  // if (printInfo) {
  //   const auto numIters = gmres_handle->get_num_iters();
  //   const auto convFlag = gmres_handle->get_conv_flag_val();
  //   const auto endRelRes = gmres_handle->get_end_rel_res();

  //   // Double check residuals at end of solve:
  //   View1D<T> r("Residual", nr);
  //   T nrmb = KokkosBlas::nrm2(b);
  //   KokkosSparse::spmv("N", 1.0, A, x, 0.0, r);
  //   KokkosBlas::axpy(-1.0, r, b);  // r = b-Ax.
  //   T endRes = KokkosBlas::nrm2(b) / nrmb;

  //   {
  //     printf("--------------------------------------------------------\n");
  //     printf("%-35s\n", (convFlag == 0) ? "GMRES: converged! :D"
  //                                       : "GMRES: not converged :(");
  //     printf("%-35s%10s\033[32m%10d\033[0m\n",
  //            "    Total Number of Iterations:", "", numIters);
  //     printf("%-35s%10s\033[32m%10.6e\033[0m\n",
  //            "    Total Reduction in Residual:", "", endRelRes);
  //     printf("%-35s%10s\033[32m%10.6e\033[0m\n",
  //            "    Compute manually |b-Ax|/|b|:", "", endRes);
  //     printf("%-35s%10s\033[32m%10.6f s\033[0m\n", "    Prepare time:", "",
  //            prepare_time);
  //     printf("%-35s%10s\033[32m%10.6f s\033[0m\n", "    Solve time:", "",
  //            solve_time);
  //     printf("--------------------------------------------------------\n");
  //   }
  // }

  // // destroy the handle
  // kh.destroy_gmres_handle();

  return Ke;
}

/*
  Compute the derivative of the stiffness matrix times the vectors psi and u
*/
template <typename T>
View1D<T> computeKDerivative(const View2D<T>& X, const View2D<int>& conn,
                             const View1D<T>& rho, const View1D<T>& u,
                             const View1D<T>& psi, const View2D<T>& C0,
                             const char* ptype_K, const double p,
                             const double q) {
  const int nelems = conn.extent(0);
  const int nnodes = X.extent(0);

  View2D<T> xe("xe", nelems, 4);
  View2D<T> ye("ye", nelems, 4);
  View2D<T> ue("ue", nelems, 8);
  View2D<T> psie("psie", nelems, 8);
  View3D<T> Be("Be", nelems, 3, 8);
  View3D<T> dfdC("dfdC", nelems, 3, 3);
  View1D<T> dK("dK", nnodes);

  // Compute Gauss quadrature with a 2-point quadrature rule
  const double gauss_pts[2] = {-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)};

  // Prepare the shape functions
  Kokkos::parallel_for(
      nelems, KOKKOS_LAMBDA(const int n) {
        for (int i = 0; i < 4; ++i) {
          xe(n, i) = X(conn(n, i), 0);
          ye(n, i) = X(conn(n, i), 1);
          ue(n, i * 2) = u(conn(n, i) * 2);
          ue(n, i * 2 + 1) = u(conn(n, i) * 2 + 1);
          psie(n, i * 2) = psi(conn(n, i) * 2);
          psie(n, i * 2 + 1) = psi(conn(n, i) * 2 + 1);
        }
      });

  for (int jj = 0; jj < 2; jj++) {
    for (int ii = 0; ii < 2; ii++) {
      T xi = gauss_pts[ii];
      T eta = gauss_pts[jj];

      auto detJ = populateBe(xi, eta, xe, ye, Be);

      // dfdC += np.einsum("n,nim,njl,nm,nl -> nij", detJ, Be, Be, psie, ue)
      Kokkos::parallel_for(
          nelems, KOKKOS_LAMBDA(const int n) {
            for (int i = 0; i < 3; i++) {
              for (int j = 0; j < 3; j++) {
                for (int m = 0; m < 8; m++) {
                  for (int l = 0; l < 8; l++) {
                    dfdC(n, i, j) += detJ(n) * Be(n, i, m) * Be(n, j, l) *
                                     psie(n, m) * ue(n, l);
                  }
                }
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
        T rhoE_n = 0.25 * (rho(conn(n, 0)) + rho(conn(n, 1)) + rho(conn(n, 2)) +
                           rho(conn(n, 3)));

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