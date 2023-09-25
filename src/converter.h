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
using View5D = Kokkos::View<T*****, Layout, ExecSpace>;
template <typename T>
using View6D = Kokkos::View<T******, Layout, ExecSpace>;

template <typename T>
using HostView1D = Kokkos::View<T*, Kokkos::HostSpace>;
template <typename T>
using HostView2D = Kokkos::View<T**, Kokkos::HostSpace>;
template <typename T>
using HostView3D = Kokkos::View<T***, Kokkos::HostSpace>;
template <typename T>
using HostView4D = Kokkos::View<T****, Kokkos::HostSpace>;
template <typename T>
using HostView5D = Kokkos::View<T*****, Kokkos::HostSpace>;
template <typename T>
using HostView6D = Kokkos::View<T******, Kokkos::HostSpace>;

#ifdef KOKKOS_ENABLE_CUDA
template <typename T>
using DeviceViewType1D = Kokkos::View<T*, Kokkos::CudaSpace>;
template <typename T>
using DeviceViewType2D = Kokkos::View<T**, Kokkos::CudaSpace>;
template <typename T>
using DeviceViewType3D = Kokkos::View<T***, Kokkos::CudaSpace>;
template <typename T>
using DeviceViewType4D = Kokkos::View<T****, Kokkos::CudaSpace>;
template <typename T>
using DeviceViewType5D = Kokkos::View<T*****, Kokkos::CudaSpace>;
template <typename T>
using DeviceViewType6D = Kokkos::View<T******, Kokkos::CudaSpace>;
#endif

void initializeKokkos() { Kokkos::initialize(); }
void finalizeKokkos() { Kokkos::finalize(); }

template <typename T>
View1D<T> numpyArrayToView1D(const py::array_t<T, py::array::c_style>& array) {
  py::buffer_info info = array.request();
  T* data_ptr = static_cast<T*>(info.ptr);

#ifdef KOKKOS_ENABLE_CUDA
  typedef DeviceViewType1D<T> ViewType;
  ViewType view("view", info.shape[0]);
  typename ViewType::HostMirror hostview(data_ptr, info.shape[0]);
  Kokkos::deep_copy(view, hostview);
#else
  // ViewType view(data_ptr, info.shape[0]);

  HostView1D<T> view("view", info.shape[0]);
  std::memcpy(view.data(), data_ptr, info.shape[0] * sizeof(T));

#endif

  return view;
}

template <typename T>
View2D<T> numpyArrayToView2D(const py::array_t<T, py::array::c_style>& array) {
  py::buffer_info info = array.request();
  T* data_ptr = static_cast<T*>(info.ptr);

#ifdef KOKKOS_ENABLE_CUDA
  typedef DeviceViewType2D<T> ViewType;
  ViewType view("view", info.shape[0], info.shape[1]);
  typename ViewType::HostMirror hostview(data_ptr, info.shape[0], info.shape[1]);
  Kokkos::deep_copy(view, hostview);
#else
  // typedef HostView2D<T> ViewType;
  // ViewType view(data_ptr, info.shape[0], info.shape[1]);
  HostView2D<T> view("view", info.shape[0], info.shape[1]);
  std::memcpy(view.data(), data_ptr, info.shape[0] * info.shape[1] * sizeof(T));
#endif

  return view;
}

template <typename T>
View3D<T> numpyArrayToView3D(const py::array_t<T, py::array::c_style>& array) {
  py::buffer_info info = array.request();
  T* data_ptr = static_cast<T*>(info.ptr);

#ifdef KOKKOS_ENABLE_CUDA
  typedef DeviceViewType3D<T> ViewType;
  ViewType view("view", info.shape[0], info.shape[1], info.shape[2]);
  typename ViewType::HostMirror hostview(data_ptr, info.shape[0], info.shape[1], info.shape[2]);
  Kokkos::deep_copy(view, hostview);
#else
  // typedef HostView3D<T> ViewType;
  // ViewType view(data_ptr, info.shape[0], info.shape[1], info.shape[2]);
  HostView3D<T> view("view", info.shape[0], info.shape[1], info.shape[2]);
  std::memcpy(view.data(), data_ptr, info.shape[0] * info.shape[1] * info.shape[2] * sizeof(T));
#endif

  return view;
}

template <typename T>
View4D<T> numpyArrayToView4D(const py::array_t<T, py::array::c_style>& array) {
  py::buffer_info info = array.request();
  T* data_ptr = static_cast<T*>(info.ptr);

#ifdef KOKKOS_ENABLE_CUDA
  typedef DeviceViewType4D<T> ViewType;
  ViewType view("view", info.shape[0], info.shape[1], info.shape[2], info.shape[3]);
  typename ViewType::HostMirror hostview(data_ptr, info.shape[0], info.shape[1], info.shape[2],
                                         info.shape[3]);
  Kokkos::deep_copy(view, hostview);
#else
  // typedef HostView4D<T> ViewType;
  // ViewType view(data_ptr, info.shape[0], info.shape[1], info.shape[2], info.shape[3]);
  HostView4D<T> view("view", info.shape[0], info.shape[1], info.shape[2], info.shape[3]);
  std::memcpy(view.data(), data_ptr,
              info.shape[0] * info.shape[1] * info.shape[2] * info.shape[3] * sizeof(T));
#endif

  return view;
}

template <typename T>
View5D<T> numpyArrayToView5D(const py::array_t<T, py::array::c_style>& array) {
  py::buffer_info info = array.request();
  T* data_ptr = static_cast<T*>(info.ptr);

#ifdef KOKKOS_ENABLE_CUDA
  typedef DeviceViewType5D<T> ViewType;
  ViewType view("view", info.shape[0], info.shape[1], info.shape[2], info.shape[3], info.shape[4]);
  typename ViewType::HostMirror hostview(data_ptr, info.shape[0], info.shape[1], info.shape[2],
                                         info.shape[3], info.shape[4]);
  Kokkos::deep_copy(view, hostview);
#else

  // typedef HostView5D<T> ViewType;
  // ViewType view(data_ptr, info.shape[0], info.shape[1], info.shape[2], info.shape[3],
  // info.shape[4]);
  HostView5D<T> view("view", info.shape[0], info.shape[1], info.shape[2], info.shape[3],
                     info.shape[4]);
  std::memcpy(
      view.data(), data_ptr,
      info.shape[0] * info.shape[1] * info.shape[2] * info.shape[3] * info.shape[4] * sizeof(T));
#endif

  return view;
}

template <typename T>
View6D<T> numpyArrayToView6D(const py::array_t<T, py::array::c_style>& array) {
  py::buffer_info info = array.request();
  T* data_ptr = static_cast<T*>(info.ptr);

#ifdef KOKKOS_ENABLE_CUDA
  typedef DeviceViewType6D<T> ViewType;
  ViewType view("view", info.shape[0], info.shape[1], info.shape[2], info.shape[3], info.shape[4],
                info.shape[5]);
  typename ViewType::HostMirror hostview(data_ptr, info.shape[0], info.shape[1], info.shape[2],
                                         info.shape[3], info.shape[4], info.shape[5]);
  Kokkos::deep_copy(view, hostview);
#else
  // typedef HostView6D<T> ViewType;
  // ViewType view(data_ptr, info.shape[0], info.shape[1], info.shape[2], info.shape[3],
  // info.shape[4], info.shape[5]);
  HostView6D<T> view("view", info.shape[0], info.shape[1], info.shape[2], info.shape[3],
                     info.shape[4], info.shape[5]);
  std::memcpy(view.data(), data_ptr,
              info.shape[0] * info.shape[1] * info.shape[2] * info.shape[3] * info.shape[4] *
                  info.shape[5] * sizeof(T));
#endif

  return view;
}

template <typename T>
py::array_t<T, py::array::c_style> viewToNumpyArray1D(const View1D<T>& view) {
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
py::array_t<T, py::array::c_style> viewToNumpyArray2D(const View2D<T>& view) {
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
py::array_t<T, py::array::c_style> viewToNumpyArray3D(const View3D<T>& view) {
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
py::array_t<T, py::array::c_style> viewToNumpyArray4D(const View4D<T>& view) {
  auto shape = view.extent(0), shape1 = view.extent(1), shape2 = view.extent(2),
       shape3 = view.extent(3);
  auto result = py::array_t<T>(shape * shape1 * shape2 * shape3);
  py::buffer_info buf = result.request();
  T* ptr = static_cast<T*>(buf.ptr);

#ifdef KOKKOS_ENABLE_CUDA
  typename View4D<T>::HostMirror hostview = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(hostview, view);
  std::memcpy(ptr, hostview.data(), shape * shape1 * shape2 * shape3 * sizeof(T));
#else
  std::memcpy(ptr, view.data(), shape * shape1 * shape2 * shape3 * sizeof(T));
#endif

  return result;
}

template <typename T>
py::array_t<T, py::array::c_style> viewToNumpyArray5D(const View5D<T>& view) {
  auto shape = view.extent(0), shape1 = view.extent(1), shape2 = view.extent(2),
       shape3 = view.extent(3), shape4 = view.extent(4);
  auto result = py::array_t<T>(shape * shape1 * shape2 * shape3 * shape4);
  py::buffer_info buf = result.request();
  T* ptr = static_cast<T*>(buf.ptr);

#ifdef KOKKOS_ENABLE_CUDA
  typename View5D<T>::HostMirror hostview = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(hostview, view);
  std::memcpy(ptr, hostview.data(), shape * shape1 * shape2 * shape3 * shape4 * sizeof(T));
#else
  std::memcpy(ptr, view.data(), shape * shape1 * shape2 * shape3 * shape4 * sizeof(T));
#endif

  return result;
}

template <typename T>
py::array_t<T, py::array::c_style> viewToNumpyArray6D(const View6D<T>& view) {
  auto shape = view.extent(0), shape1 = view.extent(1), shape2 = view.extent(2),
       shape3 = view.extent(3), shape4 = view.extent(4), shape5 = view.extent(5);
  auto result = py::array_t<T>(shape * shape1 * shape2 * shape3 * shape4 * shape5);
  py::buffer_info buf = result.request();
  T* ptr = static_cast<T*>(buf.ptr);

#ifdef KOKKOS_ENABLE_CUDA
  typename View6D<T>::HostMirror hostview = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(hostview, view);
  std::memcpy(ptr, hostview.data(), shape * shape1 * shape2 * shape3 * shape4 * shape5 * sizeof(T));
#else
  std::memcpy(ptr, view.data(), shape * shape1 * shape2 * shape3 * shape4 * shape5 * sizeof(T));
#endif

  return result;
}

#endif