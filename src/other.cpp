template <typename T>
std::vector<T> _populate_Be(int nelems, T xi, T eta,
                            const std::vector<std::vector<double>>& xe,
                            const std::vector<std::vector<double>>& ye,
                            std::vector<std::vector<std::vector<T>>>& Be) {
  std::vector<std::vector<std::vector<T>>> J(
      nelems, std::vector<std::vector<T>>(2, std::vector<T>(2, 0.0)));
  std::vector<std::vector<std::vector<T>>> invJ(
      nelems, std::vector<std::vector<T>>(2, std::vector<T>(2, 0.0)));
  std::vector<T> Nxi = {-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)};
  std::vector<T> Neta = {-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)};

  for (int i = 0; i < nelems; i++) {
    for (int m = 0; m < 2; m++) {
      for (int n = 0; n < 2; n++) {
        J[i][m][n] = xe[i][0] * Nxi[0] + xe[i][1] * Nxi[1] + xe[i][2] * Nxi[2] +
                     xe[i][3] * Nxi[3];
        J[i][m][n] += ye[i][0] * Neta[0] + ye[i][1] * Neta[1] +
                      ye[i][2] * Neta[2] + ye[i][3] * Neta[3];
      }
    }
  }

  std::vector<T> detJ(nelems, 0.0);
  for (int i = 0; i < nelems; i++) {
    detJ[i] = J[i][0][0] * J[i][1][1] - J[i][0][1] * J[i][1][0];
    invJ[i][0][0] = J[i][1][1] / detJ[i];
    invJ[i][0][1] = -J[i][0][1] / detJ[i];
    invJ[i][1][0] = -J[i][1][0] / detJ[i];
    invJ[i][1][1] = J[i][0][0] / detJ[i];
  }

  std::vector<std::vector<std::vector<T>>> Nx(
      nelems, std::vector<std::vector<T>>(4, std::vector<T>(1, 0.0)));
  std::vector<std::vector<std::vector<T>>> Ny(
      nelems, std::vector<std::vector<T>>(4, std::vector<T>(1, 0.0)));
  for (int i = 0; i < nelems; i++) {
    for (int j = 0; j < 4; j++) {
      Nx[i][j][0] = invJ[i][0][0] * Nxi[j] + invJ[i][1][0] * Neta[j];
      Ny[i][j][0] = invJ[i][0][1] * Nxi[j] + invJ[i][1][1] * Neta[j];
    }
  }

  for (int i = 0; i < nelems; ++i) {
    Be[i][0][0] = Nx[i][0][0];
    Be[i][0][2] = Nx[i][1][0];
    Be[i][0][4] = Nx[i][2][0];
    Be[i][0][6] = Nx[i][3][0];
    Be[i][1][1] = Ny[i][0][0];
    Be[i][1][3] = Ny[i][1][0];
    Be[i][1][5] = Ny[i][2][0];
    Be[i][1][7] = Ny[i][3][0];
    Be[i][2][0] = Ny[i][0][0];
    Be[i][2][2] = Ny[i][1][0];
    Be[i][2][4] = Nx[i][2][0];
    Be[i][2][6] = Nx[i][3][0];
    Be[i][2][1] = Nx[i][0][0];
    Be[i][2][3] = Nx[i][1][0];
    Be[i][2][5] = Nx[i][2][0];
    Be[i][2][7] = Nx[i][3][0];
  }
  return detJ;
}

// Function to assemble the stiffness matrix
py::array_t<double> assemble_stiffness_matrix(
    const py::array_t<double>& X, const py::array_t<int>& conn,
    const py::array_t<double>& rho, const py::array_t<double>& C0,
    double rho0_K, const std::string& ptype_K, double p, double q) {
  auto X_data = X.template unchecked<2>();
  auto conn_data = conn.template unchecked<2>();
  auto rho_data = rho.template unchecked<1>();
  auto C0_data = C0.template unchecked<2>();

  int nelems = conn.shape(0);
  int nnodes = X.shape(0);

  py::array_t<double> rhoE_array({nelems});
  auto rhoE_data = rhoE_array.mutable_unchecked<1>();

  for (int i = 0; i < nelems; ++i) {
    rhoE_data(i) =
        0.25 * (rho_data(conn_data(i, 0)) + rho_data(conn_data(i, 1)) +
                rho_data(conn_data(i, 2)) + rho_data(conn_data(i, 3)));
  }

  // // print first 10 values of rhoE
  // for (int i = 0; i < 10; ++i) {
  //   std::cout << rhoE_data(i) << std::endl;
  // }

  py::array_t<double> C_array({nelems, 9});
  auto C_data = C_array.mutable_unchecked<2>();

  if (ptype_K == "simp") {
    for (int i = 0; i < nelems; ++i) {
      for (int j = 0; j < 9; ++j) {
        C_data(i, j) = pow(rhoE_data(i), p) + rho0_K;
      }
    }
  } else {  // ramp
    for (int i = 0; i < nelems; ++i) {
      for (int j = 0; j < 9; ++j) {
        C_data(i, j) = rhoE_data(i) / (1.0 + q * (1.0 - rhoE_data(i))) + rho0_K;
      }
    }
  }

  for (int i = 0; i < nelems; ++i) {
    for (int j = 0; j < 9; ++j) {
      C_data(i, j) *= C0_data(i, j);
    }
  }

  // print first 10 values of C
  for (int i = 0; i < 10; ++i) {
    std::cout << C_data(i, 0) << std::endl;
  }

  double gauss_pts[2] = {-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)};

  py::array_t<double> Ke_array({nelems, 8, 8});
  auto Ke_data = Ke_array.mutable_unchecked<3>();

  py::array_t<double> Be_array({nelems, 3, 8});
  auto Be_data = Be_array.template mutable_unchecked<3>();

  py::array_t<double> xe_array({nelems, 4});
  auto xe_data = xe_array.mutable_unchecked<2>();

  py::array_t<double> ye_array({nelems, 4});
  auto ye_data = ye_array.mutable_unchecked<2>();

  for (int i = 0; i < nelems; i++) {
    for (int j = 0; j < 4; j++) {
      xe_data(i, j) = X_data(conn_data(i, j), 0);
      ye_data(i, j) = X_data(conn_data(i, j), 1);
    }
  }

  for (int j = 0; j < 2; j++) {
    for (int i = 0; i < 2; i++) {
      double xi = gauss_pts[i];
      double eta = gauss_pts[j];

      py::array_t<double> detJ_array =
          _populate_Be<double>(nelems, xi, eta, xe_array, ye_array, Be_array);
      auto detJ_data = detJ_array.unchecked<1>();

      for (int k = 0; k < nelems; k++) {
        py::array_t<double> CKBe_array({1, 8});
        auto CKBe_data = CKBe_array.mutable_unchecked<2>();
        for (int m = 0; m < 3; m++) {
          for (int n = 0; n < 8; n++) {
            for (int p = 0; p < 3; p++) {
              CKBe_data(0, n) += C_data(k, m * 3 + p) * Be_data(k, p, n);
            }
          }
        }
        for (int m = 0; m < 8; m++) {
          for (int n = 0; n < 8; n++) {
            Ke_data(k, m, n) +=
                detJ_data(k) * Be_data(k, n / 2, m) * CKBe_data(0, n);
          }
        }
      }
    }
  }

  return Ke_array;
}