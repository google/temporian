#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <ctime>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "temporian/implementation/numpy_cc/operators/common.h"
#include "absl/time/time.h"
namespace {
namespace py = pybind11;

absl::TimeZone handle_tz(const py::object tz) {
  if (py::isinstance<py::int_>(tz)) {
    int int_tz = tz.cast<int>();
    return absl::FixedTimeZone(int_tz * 60 * 60);
  } else if (py::isinstance<py::float_>(tz)) {
    float float_tz = tz.cast<float>();
    float_tz = float_tz * 60 * 60;
    return absl::FixedTimeZone(float_tz);
  } else if (py::isinstance<py::str>(tz)) {
    std::string str_tz = tz.cast<std::string>();
    absl::TimeZone parsed_tz;
    if (!absl::LoadTimeZone(str_tz, &parsed_tz)) {
      throw std::invalid_argument("Invalid timezone");
    }
    return parsed_tz;
    // Your C++ logic for string type
  } else {
    throw std::invalid_argument("Unsupported argument type");
  }
}

py::array_t<int> calendar_year(const py::array_t<double> &timestamps, const py::object tz) {

  auto parsed_tz = handle_tz(tz);

  // Input size
  const Idx n_events = timestamps.shape(0);

  // Allocate output array
  auto years = py::array_t<int>(n_events);

  // Access raw input / output data
  auto v_years = years.mutable_unchecked<1>();
  auto v_timestamps = timestamps.unchecked<1>();

  for (Idx i = 0; i < n_events; i++) {
    // Create tm and set the tz offset
    auto nanos = static_cast<int64_t>(v_timestamps[i] * 1e9);
    auto utc_time = absl::FromUnixNanos(nanos);
    auto local_time = absl::ToCivilDay(utc_time, parsed_tz);
    v_years[i] = local_time.year();
  }
  return years;
  }
} // namespace

void init_calendar_year(py::module &m) {
  m.def("calendar_year", &calendar_year, "", py::arg("timestamps").noconvert(), py::arg("tz"));
}