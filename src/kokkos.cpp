// #include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stiffness.h"
#include "stress_stiffness.h"
#include "wrapper.h"

namespace py = pybind11;
PYBIND11_MODULE(kokkos, m) {
  m.def("assemble_stiffness_matrix", &assembleStiffnessMatrix<double>,
        "Assemble the stiffness matrix");

  m.def("assemble_stress_stiffness", &assembleStressStiffness<double>,
        "Assemble the stress stiffness matrix");

  m.def("initialize_kokkos", &initializeKokkos, "Initialize Kokkos");

  m.def("finalize_kokkos", &finalizeKokkos, "Finalize Kokkos");
}
