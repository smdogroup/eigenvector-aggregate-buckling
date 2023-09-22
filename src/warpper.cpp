// #include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "G.h"
#include "K.h"
#include "buckling.h"
#include "converter.h"

namespace py = pybind11;

// convert the data from pyarray to kokkos view by calling the function
// convertPyArrayToView and then call the function computeElementStiffnesses
// to compute the element stiffnesses
template <typename D>
py::array_t<D> assembleK(py::array_t<D> rho_py, py::array_t<double> X_py, py::array_t<int> conn_py,
                         py::array_t<double> C0_py, double rho0_K, const char* ptype_K, double p,
                         double q) {
  auto X = numpyArrayToView2D<double>(X_py);
  auto conn = numpyArrayToView2D<int>(conn_py);
  auto rho = numpyArrayToView1D<D>(rho_py);
  auto C0 = numpyArrayToView2D<double>(C0_py);

  View3D<D> Ke = computeK<double, D>(X, conn, rho, C0, rho0_K, ptype_K, p, q);

  return viewToNumpyArray3D<D>(Ke);
}

template <typename T>
py::array_t<T> assembleKDerivative(py::array_t<T> X_py, py::array_t<int> conn_py,
                                   py::array_t<T> rho_py, py::array_t<T> u_py,
                                   py::array_t<T> psi_py, py::array_t<T> C0_py, const char* ptype_K,
                                   double p, double q) {
  auto X = numpyArrayToView2D<T>(X_py);
  auto conn = numpyArrayToView2D<int>(conn_py);
  auto rho = numpyArrayToView1D<T>(rho_py);
  auto u = numpyArrayToView1D<T>(u_py);
  auto psi = numpyArrayToView1D<T>(psi_py);
  auto C0 = numpyArrayToView2D<T>(C0_py);

  View1D<T> dK = computeKDerivative<T>(X, conn, rho, u, psi, C0, ptype_K, p, q);

  return viewToNumpyArray1D<T>(dK);
}

// convert the data from pyarray to kokkos view by calling the function
// convertPyArrayToView and then call the function computeElementStiffnesses
// to compute the element stiffnesses
template <typename D>
py::array_t<D> assembleG(py::array_t<double> X_py, py::array_t<int> conn_py, py::array_t<D> rho_py,
                         py::array_t<D> u_py, py::array_t<double> C0_py, double rho0_K,
                         const char* ptype_K, double p, double q) {
  auto X = numpyArrayToView2D<double>(X_py);
  auto conn = numpyArrayToView2D<int>(conn_py);
  auto rho = numpyArrayToView1D<D>(rho_py);
  auto u = numpyArrayToView1D<D>(u_py);
  auto C0 = numpyArrayToView2D<double>(C0_py);

  View3D<D> Ge = computeG<double, D>(X, conn, rho, u, C0, rho0_K, ptype_K, p, q);

  return viewToNumpyArray3D<D>(Ge);
}

template <typename T>
std::tuple<py::array_t<T>, py::array_t<T>, py::array_t<T>> assembleGDerivative(
    py::array_t<T> X_py, py::array_t<int> conn_py, py::array_t<T> rho_py, py::array_t<T> u_py,
    py::array_t<T> psi_py, py::array_t<T> phi_py, py::array_t<T> C0_py, T rho0_K,
    const char* ptype_K, double p, double q) {
  auto X = numpyArrayToView2D<T>(X_py);
  auto conn = numpyArrayToView2D<int>(conn_py);
  auto rho = numpyArrayToView1D<T>(rho_py);
  auto u = numpyArrayToView1D<T>(u_py);
  auto psi = numpyArrayToView1D<T>(psi_py);
  auto phi = numpyArrayToView1D<T>(phi_py);
  auto C0 = numpyArrayToView2D<T>(C0_py);

  // return dfdu and dfdC
  auto result = computeGDerivative<T>(X, conn, rho, u, psi, phi, C0, rho0_K, ptype_K, p, q);
  View1D<T> rhoE = std::get<0>(result);
  View1D<T> dfdu = std::get<1>(result);
  View3D<T> dfdC = std::get<2>(result);

  py::array_t<T> rhoE_py = viewToNumpyArray1D<T>(rhoE);
  py::array_t<T> dfdu_py = viewToNumpyArray1D<T>(dfdu);
  py::array_t<T> dfdC_py = viewToNumpyArray3D<T>(dfdC);

  return std::make_tuple(rhoE_py, dfdu_py, dfdC_py);
}

PYBIND11_MODULE(kokkos, m) {
  m.def("assemble_stiffness_matrix", &assembleK<double>, "Assemble the stiffness matrix");

  m.def("assemble_stiffness_matrix_complex", &assembleK<std::complex<double>>,
        "Assemble the stiffness matrix");

  m.def("assemble_stress_stiffness", &assembleG<double>, "Assemble the stress stiffness matrix");

  m.def("assemble_stress_stiffness_complex", &assembleG<std::complex<double>>,
        "Assemble the stress stiffness matrix");

  m.def("stiffness_matrix_derivative", &assembleKDerivative<double>,
        "Compute the derivative of the stiffness matrix");

  m.def("stress_stiffness_derivative", &assembleGDerivative<double>,
        "Compute the derivative of the stress stiffness matrix");

  m.def("initialize_kokkos", &initializeKokkos, "Initialize Kokkos");

  m.def("finalize_kokkos", &finalizeKokkos, "Finalize Kokkos");

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
}
