#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <ctime>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "temporian/implementation/numpy_cc/operators/common.h"
namespace {
namespace py = pybind11;

absl::TimeZone handle_tz(const py::object tz) {
  if (py::isinstance<py::int_>(tz)) {
    // handle tz as an int
    int int_tz = tz.cast<int>();
    // tz is expresed in hours, needs to be converted to seconds
    return absl::FixedTimeZone(int_tz * 60 * 60);

  } else if (py::isinstance<py::float_>(tz)) {
    // handle tz as an float
    float float_tz = tz.cast<float>();
    // tz is expresed in hours, needs to be converted to seconds
    return absl::FixedTimeZone(float_tz * 60 * 60);

  } else if (py::isinstance<py::str>(tz)) {
    // handle tz as an string
    std::string str_tz = tz.cast<std::string>();
    absl::TimeZone parsed_tz;
    if (!absl::LoadTimeZone(str_tz, &parsed_tz)) {
      throw std::invalid_argument("Invalid timezone '" + str_tz +
                                  "'. Only names defined in the IANA timezone" +
                                  "database are valid");
    }
    return parsed_tz;

  } else {
    throw std::invalid_argument("Unsupported argument type");
  }
}

py::array_t<int> calendar_year(const py::array_t<double> &timestamps,
                               const py::object tz) {
  auto parsed_tz = handle_tz(tz);

  // Input size
  const Idx n_events = timestamps.shape(0);

  // Allocate output array
  auto years = py::array_t<int>(n_events);

  // Access raw input / output data
  auto v_years = years.mutable_unchecked<1>();
  auto v_timestamps = timestamps.unchecked<1>();

  for (Idx i = 0; i < n_events; i++) {
    // Create absolute time in UTC
    auto nanos = static_cast<int64_t>(v_timestamps[i] * 1e9);
    auto utc_time = absl::FromUnixNanos(nanos);

    // Convert to civil time and get the year
    auto local_time = absl::ToCivilDay(utc_time, parsed_tz);
    v_years[i] = local_time.year();
  }

  return years;
}
}  // namespace

void init_calendar_year(py::module &m) {
  m.def("calendar_year", &calendar_year, "", py::arg("timestamps").noconvert(),
        py::arg("tz"));
}