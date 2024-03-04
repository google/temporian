#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cassert>
#include <cstdint>

// Index of a idx/row/column in a Numpy array.
typedef int64_t Idx;

// Converts a std::vector<T> into a Numpy array with a corresponding dtype.
//
// Usage example:
//   std::vector<double> a = {1., 2., 3.};
//   auto b = vector_to_np_array(a);
//
template <typename OutDType>
pybind11::array_t<OutDType> vector_to_np_array(
    const std::vector<OutDType>& src) {
  auto dst = pybind11::array_t<OutDType>(src.size());
  if (!src.empty()) {
    std::memcpy(dst.mutable_unchecked().mutable_data(0), &src[0],
                src.size() * sizeof(OutDType));
  }
  return dst;
}