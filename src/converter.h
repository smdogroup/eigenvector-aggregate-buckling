#ifndef CONVERTER_H
#define CONVERTER_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Kokkos_Core.hpp>

namespace py = pybind11;

#ifdef KOKKOS_ENABLE_CUDA
#define MemSpace Kokkos::CudaSpace
#endif

#ifndef MemSpace
#define MemSpace Kokkos::HostSpace
#endif

using ExecSpace = MemSpace::execution_space;
using RangePolicy = Kokkos::RangePolicy<ExecSpace>;
using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
using TeamHandle = TeamPolicy::member_type;

typedef Kokkos::DefaultExecutionSpace::array_layout Layout;

template <typename T>
using View1D = Kokkos::View<T*, Layout, ExecSpace>;
template <typename T>
using View2D = Kokkos::View<T**, Layout, ExecSpace>;
template <typename T>
using View3D = Kokkos::View<T***, Layout, ExecSpace>;
template <typename T>
using View4D = Kokkos::View<T****, Layout, ExecSpace>;

template <typename T>
using HostView1D = Kokkos::View<T*, Kokkos::HostSpace>;
template <typename T>
using HostView2D = Kokkos::View<T**, Kokkos::HostSpace>;
template <typename T>
using HostView3D = Kokkos::View<T***, Kokkos::HostSpace>;
template <typename T>
using HostView4D = Kokkos::View<T****, Kokkos::HostSpace>;

#ifdef KOKKOS_ENABLE_CUDA
template <typename T>
using DeviceViewType1D = Kokkos::View<T*, Kokkos::CudaSpace>;
template <typename T>
using DeviceViewType2D = Kokkos::View<T**, Kokkos::CudaSpace>;
template <typename T>
using DeviceViewType3D = Kokkos::View<T***, Kokkos::CudaSpace>;
template <typename T>
using DeviceViewType4D = Kokkos::View<T****, Kokkos::CudaSpace>;
#endif

void initializeKokkos() { Kokkos::initialize(); }
void finalizeKokkos() { Kokkos::finalize(); }

template <typename T>
auto numpyArrayToView1D(const py::array_t<T, py::array::c_style>& array) {
  py::buffer_info info = array.request();
  const int numDims = info.ndim;
  T* data_ptr = static_cast<T*>(info.ptr);

#ifdef KOKKOS_ENABLE_CUDA
  typedef DeviceViewType1D<T> ViewType;
  ViewType view("view", info.shape[0]);
  typename ViewType::HostMirror hostview(data_ptr, info.shape[0]);
  Kokkos::deep_copy(view, hostview);
#else
  typedef HostView1D<T> ViewType;
  ViewType view(data_ptr, info.shape[0]);
#endif

  return view;
}

template <typename T>
auto numpyArrayToView2D(const py::array_t<T, py::array::c_style>& array) {
  py::buffer_info info = array.request();
  const int numDims = info.ndim;
  T* data_ptr = static_cast<T*>(info.ptr);

#ifdef KOKKOS_ENABLE_CUDA
  typedef DeviceViewType2D<T> ViewType;
  ViewType view("view", info.shape[0], info.shape[1]);
  typename ViewType::HostMirror hostview(data_ptr, info.shape[0],
                                         info.shape[1]);
  Kokkos::deep_copy(view, hostview);
#else
  typedef HostView2D<T> ViewType;
  ViewType view(data_ptr, info.shape[0], info.shape[1]);
#endif

  return view;
}

template <typename T>
auto numpyArrayToView3D(const py::array_t<T, py::array::c_style>& array) {
  py::buffer_info info = array.request();
  const int numDims = info.ndim;
  T* data_ptr = static_cast<T*>(info.ptr);

#ifdef KOKKOS_ENABLE_CUDA
  typedef DeviceViewType3D<T> ViewType;
  ViewType view("view", info.shape[0], info.shape[1], info.shape[2]);
  typename ViewType::HostMirror hostview(data_ptr, info.shape[0], info.shape[1],
                                         info.shape[2]);
  Kokkos::deep_copy(view, hostview);
#else
  typedef HostView3D<T> ViewType;
  ViewType view(data_ptr, info.shape[0], info.shape[1], info.shape[2]);
#endif

  return view;
}

template <typename T>
auto numpyArrayToView4D(const py::array_t<T, py::array::c_style>& array) {
  py::buffer_info info = array.request();
  const int numDims = info.ndim;
  T* data_ptr = static_cast<T*>(info.ptr);

#ifdef KOKKOS_ENABLE_CUDA
  typedef DeviceViewType4D<T> ViewType;
  ViewType view("view", info.shape[0], info.shape[1], info.shape[2],
                info.shape[3]);
  typename ViewType::HostMirror hostview(data_ptr, info.shape[0], info.shape[1],
                                         info.shape[2], info.shape[3]);
  Kokkos::deep_copy(view, hostview);
#else
  typedef HostView4D<T> ViewType;
  ViewType view(data_ptr, info.shape[0], info.shape[1], info.shape[2],
                info.shape[3]);
#endif

  return view;
}

template <typename T>
auto viewToNumpyArray1D(const View1D<T>& view) {
  auto shape = view.extent(0);
  auto result = py::array_t<T>(shape);
  py::buffer_info buf = result.request();
  T* ptr = static_cast<T*>(buf.ptr);

#ifdef KOKKOS_ENABLE_CUDA
  typename View1D<T>::HostMirror hostview = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(hostview, view);
  std::memcpy(ptr, hostview.data(), shape * sizeof(T));
#else
  std::memcpy(ptr, view.data(), shape * sizeof(T));
#endif

  return result;
}

template <typename T>
auto viewToNumpyArray2D(const View2D<T>& view) {
  auto shape = view.extent(0), shape1 = view.extent(1);
  auto result = py::array_t<T>(shape * shape1);
  py::buffer_info buf = result.request();
  T* ptr = static_cast<T*>(buf.ptr);

#ifdef KOKKOS_ENABLE_CUDA
  typename View2D<T>::HostMirror hostview = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(hostview, view);
  std::memcpy(ptr, hostview.data(), shape * shape1 * sizeof(T));
#else
  std::memcpy(ptr, view.data(), shape * shape1 * sizeof(T));
#endif

  return result;
}

template <typename T>
auto viewToNumpyArray3D(const View3D<T>& view) {
  auto shape = view.extent(0), shape1 = view.extent(1), shape2 = view.extent(2);
  auto result = py::array_t<T>(shape * shape1 * shape2);
  py::buffer_info buf = result.request();
  T* ptr = static_cast<T*>(buf.ptr);

#ifdef KOKKOS_ENABLE_CUDA
  typename View3D<T>::HostMirror hostview = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(hostview, view);
  std::memcpy(ptr, hostview.data(), shape * shape1 * shape2 * sizeof(T));
#else
  std::memcpy(ptr, view.data(), shape * shape1 * shape2 * sizeof(T));
#endif

  return result;
}

template <typename T>
auto viewToNumpyArray4D(const View4D<T>& view) {
  auto shape = view.extent(0), shape1 = view.extent(1), shape2 = view.extent(2),
       shape3 = view.extent(3);
  auto result = py::array_t<T>(shape * shape1 * shape2 * shape3);
  py::buffer_info buf = result.request();
  T* ptr = static_cast<T*>(buf.ptr);

#ifdef KOKKOS_ENABLE_CUDA
  typename View4D<T>::HostMirror hostview = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(hostview, view);
  std::memcpy(ptr, hostview.data(),
              shape * shape1 * shape2 * shape3 * sizeof(T));
#else
  std::memcpy(ptr, view.data(), shape * shape1 * shape2 * shape3 * sizeof(T));
#endif

  return result;
}

#endif