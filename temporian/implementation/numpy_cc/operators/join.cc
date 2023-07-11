#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "temporian/implementation/numpy_cc/operators/common.h"

namespace {
namespace py = pybind11;

typedef int64_t OnData;

py::array_t<Idx> left_join_idxs(const py::array_t<double> &left_timestamps,
                                const py::array_t<double> &right_timestamps) {
  // Input size
  const Idx n_left = left_timestamps.shape(0);
  const Idx n_right = right_timestamps.shape(0);

  // Allocate output array
  auto idxs = py::array_t<Idx>(n_left);

  // Access raw input / output data
  auto v_idxs = idxs.mutable_unchecked<1>();
  auto v_left = left_timestamps.unchecked<1>();
  auto v_right = right_timestamps.unchecked<1>();

  Idx right_idx = 0;
  for (Idx left_idx = 0; left_idx < n_left; left_idx++) {
    const auto left = v_left[left_idx];
    while (right_idx < n_right && v_right[right_idx] < left) {
      right_idx++;
    }
    v_idxs[left_idx] =
        (right_idx < n_right && left == v_right[right_idx]) ? right_idx : -1;
  }

  return idxs;
}

py::array_t<Idx> left_join_on_idxs(const py::array_t<double> &left_timestamps,
                                   const py::array_t<double> &right_timestamps,
                                   const py::array_t<OnData> &left_on,
                                   const py::array_t<OnData> &right_on) {
  // Input size
  const Idx n_left = left_timestamps.shape(0);
  const Idx n_right = right_timestamps.shape(0);

  // Allocate output array
  auto idxs = py::array_t<Idx>(n_left);

  // Access raw input / output data
  auto v_idxs = idxs.mutable_unchecked<1>();
  auto v_left = left_timestamps.unchecked<1>();
  auto v_right = right_timestamps.unchecked<1>();
  auto v_left_on = left_on.unchecked<1>();
  auto v_right_on = right_on.unchecked<1>();

  Idx right_idx = 0;
  for (Idx left_idx = 0; left_idx < n_left; left_idx++) {
    const auto left = v_left[left_idx];
    const auto left_on = v_left_on[left_idx];

    while (right_idx < n_right && v_right[right_idx] < left) {
      right_idx++;
    }

    // Scan all the right items with the same timestamp until we find an "on"
    // match.
    auto sub_right_idx = right_idx;
    while (sub_right_idx < n_right && v_right[sub_right_idx] == left &&
           left_on != v_right_on[sub_right_idx]) {
      sub_right_idx++;
    }

    v_idxs[left_idx] =
        (sub_right_idx < n_right && left == v_right[sub_right_idx])
            ? sub_right_idx
            : -1;
  }

  return idxs;
}

}  // namespace

void init_join(py::module &m) {
  m.def("left_join_idxs", &left_join_idxs, "",
        py::arg("left_timestamps").noconvert(),
        py::arg("right_timestamps").noconvert());

  m.def("left_join_on_idxs", &left_join_on_idxs, "",
        py::arg("left_timestamps").noconvert(),
        py::arg("right_timestamps").noconvert(), py::arg("left_on").noconvert(),
        py::arg("right_on").noconvert());
}
