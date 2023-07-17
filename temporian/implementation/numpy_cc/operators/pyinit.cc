#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "temporian/implementation/numpy_cc/operators/add_index.h"
#include "temporian/implementation/numpy_cc/operators/join.h"
#include "temporian/implementation/numpy_cc/operators/resample.h"
#include "temporian/implementation/numpy_cc/operators/since_last.h"
#include "temporian/implementation/numpy_cc/operators/window.h"

namespace {
namespace py = pybind11;
}  // namespace

PYBIND11_MODULE(operators_cc, m) {
  init_since_last(m);
  init_resample(m);
  init_window(m);
  init_join(m);
  init_add_index(m);
}
