#include <assert.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <any>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "temporian/implementation/numpy_cc/operators/common.h"

namespace {
namespace py = pybind11;

// TODO: Check if absl map would be more efficient.
template <typename K, typename V>
using Map = std::unordered_map<K, V>;

/*
groups, group_indices, group_end_idx = cc_add_index(data=data)
*/

// std::tuple<py::list, py::array_t<int64_t>, py::array_t<int64_t>>
// add_index_compute_index(const py::list &data) {
//   py::print("data:", data);
//   const int num_index = data.size();
//   py::print("num_index:", num_index);

//   for (const auto &sub_data : data) {
//     if (py::isinstance<py::array_t<int64_t>>(sub_data)) {
//       const py::array_t<int64_t> sub_data_np =
//           py::cast<py::array_t<int64_t>>(sub_data);
//       py::print("dtype:", sub_data_np.dtype());
//     }
//   }

//   for (int index_idx = 0; index_idx < num_index; index_idx++) {
//     const auto &sub_data = data[index_idx];
//     py::print("sub_data:", sub_data);
//     py::print("get_type:", sub_data.get_type());
//     // py::print("isinstance array_t:",
//     py::isinstance<py::array_t>(sub_data)); py::print("isinstance
//     array_t<int64_t>:",
//               py::isinstance<py::array_t<int64_t>>(sub_data));

//     if (py::isinstance<py::array_t<int64_t>>(sub_data)) {
//       const auto sub_data_np = py::cast<py::array_t<int64_t>>(sub_data);

//       py::print("py::array_t<int64_t> dtype:", sub_data_np.dtype());
//     }

//     else if (py::isinstance<py::array>(sub_data)) {
//       const auto sub_data_np = py::cast<py::array>(sub_data);
//       print("py::array:", sub_data_np.dtype());
//       py::print("dtype itemsize:", sub_data_np.dtype().itemsize());
//       py::print("dtype kind:", sub_data_np.dtype().kind());
//       py::print("dtype num:", sub_data_np.dtype().num());

//       py::buffer_info info = sub_data_np.request();

//       // py::print("info:", info);
//       py::print("info.itemsize:", info.itemsize);
//       py::print("info.format:", info.format);
//       py::print("info.ndim:", info.ndim);
//       py::print("info.shape:", info.shape[0]);
//       py::print("info.strides:", info.strides[0]);

//       const int num_items = info.shape[0];
//       const int stride = info.strides[0];
//       for (int i = 0; i < num_items; i++) {
//         char *begin = ((char *)info.ptr) + i * stride;
//         char *end = begin + stride;
//         // py::print("begin:", begin);
//         // py::print("end:", end);

//         for (char *cur = begin; cur != end; cur++) {
//           std::cout << "D:" << (size_t)cur << "   " << (int)(*cur) << "   "
//                     << *cur << std::endl;
//         }
//       }
//     }

//     // if (py::isinstance<py::array_t<const char *>>(sub_data)) {
//     //   py::print("AAAAAAAAAAAAAAAAAAAAAAAAA");
//     // }

//     // if (py::isinstance<py::array>(sub_data)) {
//     //   py::print("BBBBBBBBBBBBBb");
//     // }

//     // else if (py::isinstance<py::array_t<std::string>>(sub_data)) {
//     //   const auto sub_data_np =
//     py::cast<py::array_t<std::string>>(sub_data);

//     //   py::print("dtype:", sub_data_np.dtype());
//     // }
//   }

//   py::list groups;

//   groups.append(1);

//   py::array_t<int64_t> indices(0);
//   py::array_t<int64_t> grouped_data_index(0);

//   // return py::make_tuple(groups, indices, grouped_data_index);
//   return std::make_tuple(groups, indices, grouped_data_index);
// }

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

  // Initializes the the group accumulator.
  //
  // Should be called once before any other method.
  void Initialize(const std::size_t num_rows) {
    assert(!initialized);
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
      assert(next_row_idx < raw_row_idxs.shape[0]);
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
void process_feature(const py::array_t<int64_t> &feature,
                     const py::list &features, const unsigned int feature_idx,
                     const std::vector<Idx> &selected_rows,
                     GroupAccumulator *index_acc,
                     std::vector<py::object> *partial_group) {
  assert(feature.ndim == 1);

  const auto raw_feature = feature.unchecked<1>();

  // List the group and the row index in each group.
  Map<int64_t, std::vector<Idx>> local_groups;
  if (selected_rows.empty()) {
    // "selected_rows" is empty i.e. we use all the rows.
    const Idx num_rows = feature.shape(0);
    if (!index_acc->initialized) {
      index_acc->Initialize(num_rows);
    }

    for (Idx row_idx = 0; row_idx < num_rows; row_idx++) {
      const int64_t value = raw_feature[row_idx];
      local_groups[value].push_back(row_idx);
    }
  } else {
    for (const auto row_idx : selected_rows) {
      const int64_t value = raw_feature[row_idx];
      local_groups[value].push_back(row_idx);
    }
  }

  // Continue building the index.
  for (auto &group : local_groups) {
    py::print("Sub group for feature_idx:", feature_idx, " with value ",
              group.first);
    partial_group->push_back(py::int_(group.first));
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
  py::print("recursive_build_index feature_idx:", feature_idx);

  if (feature_idx == features.size()) {
    index_acc->AddGroup(selected_rows, *partial_group);
    return;
  }

  const auto &feature = features[feature_idx];
  if (py::isinstance<py::array_t<int64_t>>(feature)) {
    const auto &casted_feature = py::cast<py::array_t<int64_t>>(feature);
    process_feature(casted_feature, features, feature_idx, selected_rows,
                    index_acc, partial_group);
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

}  // namespace

void init_add_index(py::module &m) {
  m.def("add_index_compute_index", &add_index_compute_index, "",
        py::arg("features").noconvert());
}
