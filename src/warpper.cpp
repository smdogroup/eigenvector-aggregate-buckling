// #include <omp.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "G.h"
#include "K.h"
#include "buckling.h"
#include "converter.h"
#include "lobpcg.hpp"

namespace py = pybind11;

template <typename T>
std::tuple<py::array_t<T>, py::array_t<T>> populate_Be(T xi, T eta, py::array_t<T> xe_py,
                                                       py::array_t<T> ye_py) {
  auto xe = numpyArrayToView2D<T>(xe_py);
  auto ye = numpyArrayToView2D<T>(ye_py);
  View3D<T> Be("Be", xe.extent(0), 3, 8);

  View1D<T> detJ = populateBe<T>(xi, eta, xe, ye, Be);

  auto detJ_py = viewToNumpyArray1D<T>(detJ);
  auto Be_py = viewToNumpyArray3D<T>(Be);

  return std::make_tuple(detJ_py, Be_py);
}

template <typename T>
std::tuple<py::array_t<T>, py::array_t<T>, py::array_t<T>> populate_Be_Te(T xi, T eta,
                                                                          py::array_t<T> xe_py,
                                                                          py::array_t<T> ye_py) {
  auto xe = numpyArrayToView2D<T>(xe_py);
  auto ye = numpyArrayToView2D<T>(ye_py);
  View3D<T> Be("Be", xe.extent(0), 3, 8);
  View4D<T> Te("Te", xe.extent(0), 3, 4, 4);

  View1D<T> detJ = populateBeTe<T>(xi, eta, xe, ye, Be, Te);

  auto detJ_py = viewToNumpyArray1D<T>(detJ);
  auto Be_py = viewToNumpyArray3D<T>(Be);
  auto Te_py = viewToNumpyArray4D<T>(Te);

  return std::make_tuple(detJ_py, Be_py, Te_py);
}

// convert the data from pyarray to kokkos view by calling the function
// convertPyArrayToView and then call the function computeElementStiffnesses
// to compute the element stiffnesses
template <typename D>
py::array_t<D> assembleK(py::array_t<D> rho_py, py::array_t<double> detJ_py,
                         py::array_t<double> Be_py, py::array_t<int> conn_py,
                         py::array_t<double> C0_py, double rho0_K, const char* ptype_K, double p,
                         double q) {
  auto rho = numpyArrayToView1D<D>(rho_py);
  auto detJ = numpyArrayToView3D<double>(detJ_py);
  auto Be = numpyArrayToView5D<double>(Be_py);
  auto conn = numpyArrayToView2D<int>(conn_py);
  auto C0 = numpyArrayToView2D<double>(C0_py);

  View3D<D> Ke = computeK<double, D>(rho, detJ, Be, conn, C0, rho0_K, ptype_K, p, q);

  return viewToNumpyArray3D<D>(Ke);
}

// Wrapper function that dispatches to the appropriate add function
// based on the data types for rho_py
py::object assembleK_generic(py::object rho_py, py::array_t<double> detJ_py,
                             py::array_t<double> Be_py, py::array_t<int> conn_py,
                             py::array_t<double> C0_py, double rho0_K, const char* ptype_K,
                             double p, double q) {
  if (py::isinstance<py::array_t<double>>(rho_py)) {
    return assembleK<double>(rho_py.cast<py::array_t<double>>(), detJ_py, Be_py, conn_py, C0_py,
                             rho0_K, ptype_K, p, q);
  } else if (py::isinstance<py::array_t<std::complex<double>>>(rho_py)) {
    return assembleK<std::complex<double>>(rho_py.cast<py::array_t<std::complex<double>>>(),
                                           detJ_py, Be_py, conn_py, C0_py, rho0_K, ptype_K, p, q);
  } else {
    throw std::runtime_error("Unsupported data type for rho_py");
  }
}

template <typename T>
py::array_t<T> assembleKDerivative(py::array_t<T> rho_py, py::array_t<T> detJ_py,
                                   py::array_t<T> Be_py, py::array_t<int> conn_py,
                                   py::array_t<T> u_py, py::array_t<T> psi_py, py::array_t<T> C0_py,
                                   const char* ptype_K, double p, double q) {
  auto rho = numpyArrayToView1D<T>(rho_py);
  auto detJ = numpyArrayToView3D<T>(detJ_py);
  auto Be = numpyArrayToView5D<T>(Be_py);
  auto conn = numpyArrayToView2D<int>(conn_py);
  auto u = numpyArrayToView1D<T>(u_py);
  auto psi = numpyArrayToView1D<T>(psi_py);
  auto C0 = numpyArrayToView2D<T>(C0_py);

  View1D<T> dK = computeKDerivative<T>(rho, detJ, Be, conn, u, psi, C0, ptype_K, p, q);

  return viewToNumpyArray1D<T>(dK);
}

// convert the data from pyarray to kokkos view by calling the function
// convertPyArrayToView and then call the function computeElementStiffnesses
// to compute the element stiffnesses
template <typename D>
py::array_t<D> assembleG(py::array_t<D> rho_py, py::array_t<D> u_py, py::array_t<double> detJ_py,
                         py::array_t<double> Be_py, py::array_t<double> Te_py,
                         py::array_t<int> conn_py, py::array_t<double> C0_py, double rho0_K,
                         const char* ptype_K, double p, double q) {
  auto rho = numpyArrayToView1D<D>(rho_py);
  auto u = numpyArrayToView1D<D>(u_py);
  auto detJ = numpyArrayToView3D<double>(detJ_py);
  auto Be = numpyArrayToView5D<double>(Be_py);
  auto Te = numpyArrayToView6D<double>(Te_py);
  auto conn = numpyArrayToView2D<int>(conn_py);
  auto C0 = numpyArrayToView2D<double>(C0_py);

  View3D<D> Ge = computeG<double, D>(rho, u, detJ, Be, Te, conn, C0, rho0_K, ptype_K, p, q);

  return viewToNumpyArray3D<D>(Ge);
}

// Wrapper function that dispatches to the appropriate add function
// based on the data types for rho_py
py::object assembleG_generic(py::object rho_py, py::object u_py, py::array_t<double> detJ_py,
                             py::array_t<double> Be_py, py::array_t<double> Te_py,
                             py::array_t<int> conn_py, py::array_t<double> C0_py, double rho0_K,
                             const char* ptype_K, double p, double q) {
  if (py::isinstance<py::array_t<double>>(rho_py)) {
    return assembleG<double>(rho_py.cast<py::array_t<double>>(), u_py.cast<py::array_t<double>>(),
                             detJ_py, Be_py, Te_py, conn_py, C0_py, rho0_K, ptype_K, p, q);
  } else if (py::isinstance<py::array_t<std::complex<double>>>(rho_py)) {
    return assembleG<std::complex<double>>(rho_py.cast<py::array_t<std::complex<double>>>(),
                                           u_py.cast<py::array_t<std::complex<double>>>(), detJ_py,
                                           Be_py, Te_py, conn_py, C0_py, rho0_K, ptype_K, p, q);
  } else {
    throw std::runtime_error("Unsupported data type for rho_py");
  }
}

template <typename T>
std::tuple<py::array_t<T>, py::array_t<T>, py::array_t<T>> assembleGDerivative(
    py::array_t<T> rho_py, py::array_t<T> u_py, py::array_t<T> detJ_py, py::array_t<T> Be_py,
    py::array_t<T> Te_py, py::array_t<int> conn_py, py::array_t<T> psi_py, py::array_t<T> phi_py,
    py::array_t<T> C0_py, T rho0_K, const char* ptype_K, double p, double q) {
  auto rho = numpyArrayToView1D<T>(rho_py);
  auto u = numpyArrayToView1D<T>(u_py);
  auto detJ = numpyArrayToView3D<T>(detJ_py);
  auto Be = numpyArrayToView5D<T>(Be_py);
  auto Te = numpyArrayToView6D<T>(Te_py);
  auto conn = numpyArrayToView2D<int>(conn_py);
  auto psi = numpyArrayToView1D<T>(psi_py);
  auto phi = numpyArrayToView1D<T>(phi_py);
  auto C0 = numpyArrayToView2D<T>(C0_py);

  // return dfdu and dfdC
  auto result =
      computeGDerivative<T>(rho, u, detJ, Be, Te, conn, psi, phi, C0, rho0_K, ptype_K, p, q);

  View1D<T> rhoE = std::get<0>(result);
  View1D<T> dfdu = std::get<1>(result);
  View3D<T> dfdC = std::get<2>(result);

  py::array_t<T> rhoE_py = viewToNumpyArray1D<T>(rhoE);
  py::array_t<T> dfdu_py = viewToNumpyArray1D<T>(dfdu);
  py::array_t<T> dfdC_py = viewToNumpyArray3D<T>(dfdC);

  return std::make_tuple(rhoE_py, dfdu_py, dfdC_py);
}

template <typename T, typename I>
std::tuple<py::array_t<T>, py::array_t<T>> py_lobpcg(py::array_t<T> Ax, py::array_t<I> Ap,
                                                     py::array_t<I> Aj, py::array_t<T> Bx,
                                                     py::array_t<I> Bp, py::array_t<I> Bj, int n,
                                                     int m, py::array_t<T> Mp) {
  // Extract data pointers from NumPy arrays
  T* Ax_ptr = Ax.mutable_data();
  I* Ap_ptr = Ap.mutable_data();
  I* Aj_ptr = Aj.mutable_data();
  T* Bx_ptr = Bx.mutable_data();
  I* Bp_ptr = Bp.mutable_data();
  I* Bj_ptr = Bj.mutable_data();
  T* Mp_ptr = Mp.mutable_data();

  // create output arrays
  py::array_t<T> wp_py = py::array_t<T>(m);
  py::array_t<T> vp_py = py::array_t<T>(n * m);

  // Extract data pointers from NumPy arrays
  T* wp = wp_py.mutable_data();
  T* vp = vp_py.mutable_data();

  // call lobpcg function
  linalg::sparse::lobpcg<T, I>(Ax_ptr, Ap_ptr, Aj_ptr, Bx_ptr, Bp_ptr, Bj_ptr, n, m, wp, vp, Mp_ptr);

  return std::make_tuple(wp_py, vp_py);
}

PYBIND11_MODULE(kokkos, m) {
  m.def("populate_Be", &populate_Be<double>, "Populate the Be matrix");

  m.def("populate_Be_Te", &populate_Be_Te<double>, "Populate the Be and Te matrices");

  m.def("assemble_stiffness_matrix", &assembleK_generic, "Assemble the stiffness matrix");

  m.def("assemble_stress_stiffness", &assembleG_generic, "Assemble the stress stiffness matrix");

  m.def("stiffness_matrix_derivative", &assembleKDerivative<double>,
        "Compute the derivative of the stiffness matrix");

  m.def("stress_stiffness_derivative", &assembleGDerivative<double>,
        "Compute the derivative of the stress stiffness matrix");

  m.def("lobpcg", &py_lobpcg<double, int>, "Run lobpcg function");

  m.def("initialize", &initializeKokkos, "Initialize Kokkos");

  m.def("finalize", &finalizeKokkos, "Finalize Kokkos");
}

// py::class_<Buckling<double>>(m, "Buckling")
//     .def(py::init(
//         [](py::array_t<double> X_py, py::array_t<int> conn_py, py::array_t<double> rho_py,
//            py::array_t<double> u_py, py::array_t<double> psi_py, py::array_t<double> phi_py,
//            py::array_t<double> C0_py, double rho0_K, std::string ptype_K, double p, double q) {
//           // Convert numpy arrays to Kokkos views
//           auto X = numpyArrayToView2D<double>(X_py);
//           auto conn = numpyArrayToView2D<int>(conn_py);
//           auto rho = numpyArrayToView1D<double>(rho_py);
//           auto u = numpyArrayToView1D<double>(u_py);
//           auto psi = numpyArrayToView1D<double>(psi_py);
//           auto phi = numpyArrayToView1D<double>(phi_py);
//           auto C0 = numpyArrayToView2D<double>(C0_py);

//           // Create the Buckling object
//           return new Buckling<double>(X, conn, rho, u, psi, phi, C0, rho0_K, ptype_K, p, q);
//         }))
//     .def("computeK",
//          [](Buckling<double>& self) {
//            View3D<double> K = self.computeK();
//            return viewToNumpyArray3D(K);
//          })
//     .def("computeG",
//          [](Buckling<double>& self) {
//            View3D<double> G = self.computeG();
//            return viewToNumpyArray3D(G);
//          })
//     .def("computeKDerivative",
//          [](Buckling<double>& self) {
//            View1D<double> dK = self.computeKDerivative();
//            return viewToNumpyArray1D(dK);
//          })
//     .def("computeGDerivative", [](Buckling<double>& self) {
//       View1D<double> dG = self.computeGDerivative();
//       return viewToNumpyArray1D(dG);
//     });
