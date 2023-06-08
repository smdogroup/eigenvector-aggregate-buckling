#include <pybind11/pybind11.h>

// #include <unsupported/Eigen/MPRealSupport>

#include 
#include "mpreal.h"
// using namespace mpfr;
namespace py = pybind11;
using mpfr::mpreal;
double precise(double rho, double trace, double lam_min, double lam1,
               double lam2) {
  mpfr::mpreal::set_default_prec(80);

  mpfr::mpreal val;
  if (lam1 == lam2) {
    val = -rho * mpreal::exp(-rho * (lam1 - lam_min)) / trace;
  } else {
    val = (mpreal::exp(-rho * (lam1 - lam_min)) -
           mpreal::exp(-rho * (lam2 - lam_min))) /
          (mpreal(lam1) - mpreal(lam2)) / mpreal(trace);
  }

  return val.toDouble();
}

PYBIND11_MODULE(eij, m) {
  m.doc() = "Precise module";

  m.def("precise", &precise, "Calculate the precise value.");
}