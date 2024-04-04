#include <assert.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <any>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "temporian/implementation/numpy_cc/operators/common.h"

namespace {
namespace py = pybind11;

// TODO: Check if absl map would be more efficient.
template <typename K, typename V> using Map = std::unordered_map<K, V>;

// Aggregates group data to be returned.
struct GroupAccumulator {
  // Was the struct initialized i.e. was "Initialize" called?
  bool initialized = false;

  // List of list of feature values defining group keys.
  // group_keys[i][j] is the j-th index key value of the i-th group.
  //
  // At the end of the accumulation, "group_keys.size()" is the number of
  // groups.
  py::list group_keys;

  // List of example indices. Contains the same number of values as the number
  // of input row. See "group_indices".
  py::array_t<Idx> row_idxs;

  // Indirect row index of the groups.
  //
  // row_idxs[group_begin_idx[i]]..row_idxs[group_begin_idx[i+1]] are the
  // indices of the i-th group.
  //
  // At the end of the accumulation, contains g+1 values where g is the number
  // of groups.
  std::vector<Idx> group_begin_idx;

  // Index of the next available spot in "row_idxs".
  Idx next_row_idx = 0;

  // Initializes the group accumulator.
  //
  // Should be called once before any other method.
  void Initialize(const std::size_t num_rows) {
    assert(!initialized);
    initialized = true;
    group_begin_idx.push_back(0);
    row_idxs = py::array_t<Idx>(num_rows);
  }

  py::array_t<Idx> GroupBeginIdxToArray() {
    py::array_t<Idx> result(group_begin_idx.size());
    std::copy(group_begin_idx.begin(), group_begin_idx.end(),
              result.mutable_data());
    return result;
  }

  // Adds a group.
  //
  // Args:
  //   selected_rows: Rows in this group.
  //   group_key: The key of this group.
  //
  void AddGroup(const std::vector<Idx> &selected_rows,
                const std::vector<py::object> &group_key) {
    assert(initialized);

    // Copy the group indices
    auto raw_row_idxs = row_idxs.mutable_unchecked<1>();
    for (const Idx selected_row : selected_rows) {
      // assert(next_row_idx < raw_row_idxs.shape[0]);
      raw_row_idxs[next_row_idx++] = selected_row;
    }

    // Record the position of the last row idx in the group.
    group_begin_idx.push_back(next_row_idx);

    // Convert the group key into a python list.
    py::list py_group_key;
    for (const auto &key : group_key) {
      py_group_key.append(key);
    }
    group_keys.append(py::tuple(py_group_key));
  }
};

// Forward declaration
void recursive_build_index(const py::list &features,
                           const unsigned int feature_idx,
                           const std::vector<Idx> &selected_rows,
                           GroupAccumulator *index_acc,
                           std::vector<py::object> *partial_group);

// Builds the index of a given feature and recursively call
// "recursive_build_index" on the remaining features.
template <typename Feature>
void process_feature_int(const py::array_t<Feature> &feature,
                         const py::list &features,
                         const unsigned int feature_idx,
                         const std::vector<Idx> &selected_rows,
                         GroupAccumulator *index_acc,
                         std::vector<py::object> *partial_group) {
  assert(feature.ndim() == 1);

  const auto raw_feature = feature.template unchecked<1>();

  // List the group and the row index in each group.
  Map<Feature, std::vector<Idx>> local_groups;
  if (selected_rows.empty()) {
    // "selected_rows" is empty i.e. we use all the rows.
    const Idx num_rows = feature.shape(0);
    if (!index_acc->initialized) {
      index_acc->Initialize(num_rows);
    }

    for (Idx row_idx = 0; row_idx < num_rows; row_idx++) {
      const Feature value = raw_feature[row_idx];
      local_groups[value].push_back(row_idx);
    }
  } else {
    for (const auto row_idx : selected_rows) {
      const Feature value = raw_feature[row_idx];
      local_groups[value].push_back(row_idx);
    }
  }

  // Continue building the index.
  for (auto &group : local_groups) {
    partial_group->push_back(py::int_(group.first));
    recursive_build_index(features, feature_idx + 1, group.second, index_acc,
                          partial_group);
    partial_group->pop_back();
  }
}

std::string_view remove_tailing_zeros(std::string_view src) {
  int i = static_cast<int>(src.size()) - 1;
  while (i >= 0 && src[i] == 0) {
    i--;
  }
  return src.substr(0, i + 1);
}

// Builds the index of a given feature and recursively call
// "recursive_build_index" on the remaining features.
void process_feature_string(const py::array &feature, const py::list &features,
                            const unsigned int feature_idx,
                            const std::vector<Idx> &selected_rows,
                            GroupAccumulator *index_acc,
                            std::vector<py::object> *partial_group) {
  assert(feature.ndim() == 1);
  py::buffer_info info = feature.request();

  // List the group and the row index in each group.
  Map<std::string_view, std::vector<Idx>> local_groups;
  const int stride = info.strides[0];
  const int itemsize = info.itemsize;
  if (selected_rows.empty()) {
    // "selected_rows" is empty i.e. we use all the rows.
    const Idx num_rows = info.shape[0];
    if (!index_acc->initialized) {
      index_acc->Initialize(num_rows);
    }
    for (Idx row_idx = 0; row_idx < num_rows; row_idx++) {
      const char *begin = ((char *)info.ptr) + row_idx * stride;
      std::string_view value(begin, itemsize);
      local_groups[remove_tailing_zeros(value)].push_back(row_idx);
    }
  } else {
    for (const auto row_idx : selected_rows) {
      const char *begin = ((char *)info.ptr) + row_idx * stride;
      std::string_view value(begin, itemsize);
      local_groups[remove_tailing_zeros(value)].push_back(row_idx);
    }
  }

  // Continue building the index.
  for (auto &group : local_groups) {
    partial_group->push_back(py::bytes(group.first));
    recursive_build_index(features, feature_idx + 1, group.second, index_acc,
                          partial_group);
    partial_group->pop_back();
  }
}

void recursive_build_index(const py::list &features,
                           const unsigned int feature_idx,
                           const std::vector<Idx> &selected_rows,
                           GroupAccumulator *index_acc,
                           std::vector<py::object> *partial_group) {
  if (feature_idx == features.size()) {
    index_acc->AddGroup(selected_rows, *partial_group);
    return;
  }

  const auto &feature = features[feature_idx];

  if (py::isinstance<py::array_t<int64_t>>(feature)) {
    const auto &casted_feature = py::cast<py::array_t<int64_t>>(feature);
    process_feature_int<int64_t>(casted_feature, features, feature_idx,
                                 selected_rows, index_acc, partial_group);
    return;
  }

  if (py::isinstance<py::array_t<int32_t>>(feature)) {
    const auto &casted_feature = py::cast<py::array_t<int32_t>>(feature);
    process_feature_int<int32_t>(casted_feature, features, feature_idx,
                                 selected_rows, index_acc, partial_group);
    return;
  }

  if (py::isinstance<py::array>(feature)) {
    const auto casted_feature = py::cast<py::array>(feature);
    if (casted_feature.dtype().kind() == 'S') {
      process_feature_string(casted_feature, features, feature_idx,
                             selected_rows, index_acc, partial_group);
    }
    return;
  }

  py::print("Feature:", feature.get_type());
  throw std::invalid_argument("Non supported feature type");
}

// Computes the groups and row idxs in groups.
//
// Args:
//   features: List of numpy array containing the features to index on.
std::tuple<py::list, py::array_t<Idx>, py::array_t<Idx>>
add_index_compute_index(const py::list &features) {
  GroupAccumulator index_acc;
  std::vector<py::object> partial_group;
  recursive_build_index(features, /*feature_idx=*/0, /*selected_rows=*/{},
                        &index_acc, &partial_group);

  return std::make_tuple(index_acc.group_keys, index_acc.row_idxs,
                         index_acc.GroupBeginIdxToArray());
}

} // namespace

void init_add_index(py::module &m) {
  m.def("add_index_compute_index", &add_index_compute_index, "",
        py::arg("features").noconvert());
}
