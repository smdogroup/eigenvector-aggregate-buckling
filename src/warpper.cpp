// #include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "G.h"
#include "K.h"
#include "buckling.h"
#include "converter.h"

namespace py = pybind11;
PYBIND11_MODULE(kokkos, m) {
  m.def("assemble_stiffness_matrix", &assembleK<double>,
        "Assemble the stiffness matrix");

  m.def("assemble_stress_stiffness", &assembleG<double>,
        "Assemble the stress stiffness matrix");

  m.def("stiffness_matrix_derivative", &assembleKDerivative<double>,
        "Compute the derivative of the stiffness matrix");

  m.def("initialize_kokkos", &initializeKokkos, "Initialize Kokkos");

  m.def("finalize_kokkos", &finalizeKokkos, "Finalize Kokkos");

  py::class_<Buckling<double>>(m, "Buckling")
      .def(py::init([](py::array_t<double> X_py, py::array_t<int> conn_py,
                       py::array_t<double> rho_py, py::array_t<double> u_py,
                       py::array_t<double> psi_py, py::array_t<double> phi_py,
                       py::array_t<double> C0_py, double rho0_K,
                       std::string ptype_K, double p, double q) {
        // Convert numpy arrays to Kokkos views
        auto X = numpyArrayToView2D<double>(X_py);
        auto conn = numpyArrayToView2D<int>(conn_py);
        auto rho = numpyArrayToView1D<double>(rho_py);
        auto u = numpyArrayToView1D<double>(u_py);
        auto psi = numpyArrayToView1D<double>(psi_py);
        auto phi = numpyArrayToView1D<double>(phi_py);
        auto C0 = numpyArrayToView2D<double>(C0_py);

        // Create the Buckling object
        return new Buckling<double>(X, conn, rho, u, psi, phi, C0, rho0_K,
                                    ptype_K, p, q);
      }))
      .def("computeK",
           [](Buckling<double>& self) {
             View3D<double> K = self.computeK();
             return viewToNumpyArray3D(K);
           })
      .def("computeG",
           [](Buckling<double>& self) {
             View3D<double> G = self.computeG();
             return viewToNumpyArray3D(G);
           })
      .def("computeKDerivative",
           [](Buckling<double>& self) {
             View1D<double> dK = self.computeKDerivative();
             return viewToNumpyArray1D(dK);
           })
      .def("computeGDerivative", [](Buckling<double>& self) {
        View1D<double> dG = self.computeGDerivative();
        return viewToNumpyArray1D(dG);
      });
}
