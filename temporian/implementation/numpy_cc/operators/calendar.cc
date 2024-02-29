#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

#include "absl/time/time.h"
#include "temporian/implementation/numpy_cc/operators/common.h"
#include "temporian/implementation/numpy_cc/operators/tick_calendar_utils.h"

namespace {
namespace py = pybind11;

// Function for iterating a timestamps array, converting to civil time
// and apply a calendar_op to each one
ErrorString apply_calendar_op(const py::array_t<double> &timestamps,
                              const py::object tz,
                              std::function<int(absl::CivilSecond)> calendar_op,
                              py::array_t<int32_t> &output) {
  absl::TimeZone parsed_tz;
  const auto error = parse_tz(tz, parsed_tz);
  if (error.has_value()) {
    return error;
  }

  // Input size
  const Idx n_events = timestamps.shape(0);

  // Access raw input / output data
  auto v_output = output.mutable_unchecked<1>();
  auto v_timestamps = timestamps.unchecked<1>();

  for (Idx i = 0; i < n_events; i++) {
    // Create absolute time in UTC
    const auto nanos = static_cast<int64_t>(v_timestamps[i] * 1e9);
    const auto utc_time = absl::FromUnixNanos(nanos);

    // Convert to civil time and call calendar_op
    const auto local_time = absl::ToCivilSecond(utc_time, parsed_tz);
    v_output[i] = calendar_op(local_time);
  }

  return {};
}

// year calendar op
ErrorString calendar_year(const py::array_t<double> &timestamps,
                          const py::object tz, py::array_t<int> &output) {
  return apply_calendar_op(
      timestamps, tz, [](absl::CivilSecond value) { return value.year(); },
      output);
}

// month calendar op
ErrorString calendar_month(const py::array_t<double> &timestamps,
                           const py::object tz, py::array_t<int> &output) {
  return apply_calendar_op(
      timestamps, tz, [](absl::CivilSecond value) { return value.month(); },
      output);
}

// hour calendar op
ErrorString calendar_hour(const py::array_t<double> &timestamps,
                          const py::object tz, py::array_t<int> &output) {
  return apply_calendar_op(
      timestamps, tz, [](absl::CivilSecond value) { return value.hour(); },
      output);
}

// day of month calendar op
ErrorString calendar_day_of_month(const py::array_t<double> &timestamps,
                                  const py::object tz,
                                  py::array_t<int> &output) {
  return apply_calendar_op(
      timestamps, tz, [](absl::CivilSecond value) { return value.day(); },
      output);
}

// minute calendar op
ErrorString calendar_minute(const py::array_t<double> &timestamps,
                            const py::object tz, py::array_t<int> &output) {
  return apply_calendar_op(
      timestamps, tz, [](absl::CivilSecond value) { return value.minute(); },
      output);
}

// second calendar op
ErrorString calendar_second(const py::array_t<double> &timestamps,
                            const py::object tz, py::array_t<int> &output) {
  return apply_calendar_op(
      timestamps, tz, [](absl::CivilSecond value) { return value.second(); },
      output);
}

// day of year calendar op
ErrorString calendar_day_of_year(const py::array_t<double> &timestamps,
                                 const py::object tz,
                                 py::array_t<int> &output) {
  return apply_calendar_op(
      timestamps, tz,
      [](absl::CivilSecond value) { return absl::GetYearDay(value); }, output);
}

// day of week calendar op
ErrorString calendar_day_of_week(const py::array_t<double> &timestamps,
                                 const py::object tz,
                                 py::array_t<int> &output) {
  return apply_calendar_op(
      timestamps, tz,
      [](absl::CivilSecond value) {
        return map_week_day(absl::GetWeekday(value));
      },
      output);
}

// isoweek
ErrorString calendar_isoweek(const py::array_t<double> &timestamps,
                             const py::object tz, py::array_t<int> &output) {
  auto iso_week = [](absl::CivilSecond value) {
    // directly translated to cpp from panda's ccalendar.pyx implementation
    // https://github.com/pandas-dev/pandas/blob/1c606d5f014c5296d6028af28001311b67ee3721/pandas/_libs/tslibs/ccalendar.pyx

    auto doy = absl::GetYearDay(value);
    auto dow = map_week_day(absl::GetWeekday(value));
    auto day = value.day();
    auto year = value.year();
    // estimate
    auto iso_week = (doy - 1) - dow + 3;
    if (iso_week >= 0) {
      iso_week = iso_week / 7 + 1;
    }
    // verify
    if (iso_week < 0) {
      if ((iso_week > -2) or (iso_week == -2 and IsLeapYear(year - 1))) {
        iso_week = 53;
      } else {
        iso_week = 52;
      }
    } else if (iso_week == 53) {
      if (31 - day + dow < 3) {
        iso_week = 1;
      }
    }
    return iso_week;
  };
  return apply_calendar_op(timestamps, tz, iso_week, output);
}

}  // namespace

void init_calendar_ops(py::module &m) {
  m.def("calendar_year", &calendar_year, "", py::arg("timestamps").noconvert(),
        py::arg("tz").noconvert(), py::arg("output").noconvert());
  m.def("calendar_month", &calendar_month, "",
        py::arg("timestamps").noconvert(), py::arg("tz").noconvert(),
        py::arg("output").noconvert());

  m.def("calendar_day_of_month", &calendar_day_of_month, "",
        py::arg("timestamps").noconvert(), py::arg("tz").noconvert(),
        py::arg("output").noconvert());

  m.def("calendar_hour", &calendar_hour, "", py::arg("timestamps").noconvert(),
        py::arg("tz").noconvert(), py::arg("output").noconvert());

  m.def("calendar_minute", &calendar_minute, "",
        py::arg("timestamps").noconvert(), py::arg("tz").noconvert(),
        py::arg("output").noconvert());

  m.def("calendar_second", &calendar_second, "",
        py::arg("timestamps").noconvert(), py::arg("tz").noconvert(),
        py::arg("output").noconvert());

  m.def("calendar_day_of_year", &calendar_day_of_year, "",
        py::arg("timestamps").noconvert(), py::arg("tz").noconvert(),
        py::arg("output").noconvert());

  m.def("calendar_day_of_week", &calendar_day_of_week, "",
        py::arg("timestamps").noconvert(), py::arg("tz").noconvert(),
        py::arg("output").noconvert());

  m.def("calendar_isoweek", &calendar_isoweek, "",
        py::arg("timestamps").noconvert(), py::arg("tz").noconvert(),
        py::arg("output"));
}