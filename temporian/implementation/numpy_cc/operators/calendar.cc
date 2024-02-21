#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>

#include "absl/time/time.h"
#include "temporian/implementation/numpy_cc/operators/common.h"

namespace {
namespace py = pybind11;

std::string parse_tz(const py::object &tz, absl::TimeZone &parsed_tz) {
  if (py::isinstance<py::int_>(tz)) {
    // handle tz as an int
    const int int_tz = tz.cast<int>();
    // tz is expresed in hours, needs to be converted to seconds
    parsed_tz = absl::FixedTimeZone(int_tz * 60 * 60);

  } else if (py::isinstance<py::float_>(tz)) {
    // handle tz as an float
    const float float_tz = tz.cast<float>() * 60 * 60;
    const int int_tz = static_cast<int>(std::round(float_tz));
    // tz is expresed in hours, needs to be converted to seconds
    parsed_tz = absl::FixedTimeZone(int_tz);

  } else if (py::isinstance<py::str>(tz)) {
    // handle tz as an string
    const std::string str_tz = tz.cast<std::string>();
    if (!absl::LoadTimeZone(str_tz, &parsed_tz)) {
      return "Invalid timezone '" + str_tz +
             "'. Only names defined in the IANA timezone" +
             "database are valid";
    }

  } else {
    return "Unsupported argument type for \"tz\" argument. Expecting int, "
           "float, "
           "or str";
  }

  return "";
}

int is_leapyear(int64_t year) {
  return ((year & 0x3) == 0 and  // year % 4 == 0
          ((year % 100) != 0 or (year % 400) == 0));
}

// TODO: support week starting on Sunday
int map_week_day(const absl::Weekday &wd) {
  switch (wd) {
    case absl::Weekday::monday:
      return 0;
    case absl::Weekday::tuesday:
      return 1;
    case absl::Weekday::wednesday:
      return 2;
    case absl::Weekday::thursday:
      return 3;
    case absl::Weekday::friday:
      return 4;
    case absl::Weekday::saturday:
      return 5;
    case absl::Weekday::sunday:
      return 6;
  }
}

// Function for iterating a timestamps array, converting to civil time
// and apply a calendar_op to each one
std::string apply_calendar_op(const py::array_t<double> &timestamps,
                              const py::object tz,
                              std::function<int(absl::CivilSecond)> calendar_op,
                              py::array_t<int32_t> &output) {
  absl::TimeZone parsed_tz;
  const auto error = parse_tz(tz, parsed_tz);
  if (!error.empty()) {
    return error;
  }

  // Input size
  const Idx n_events = timestamps.shape(0);

  // Allocate output array
  // auto output = py::array_t<int>(n_events);

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

  return "";
}

// year calendar op
int year(absl::CivilSecond value) { return value.year(); }

std::string calendar_year(const py::array_t<double> &timestamps,
                          const py::object tz, py::array_t<int> &output) {
  return apply_calendar_op(timestamps, tz, year, output);
}

// month calendar op
int month(absl::CivilSecond value) { return value.month(); }

std::string calendar_month(const py::array_t<double> &timestamps,
                           const py::object tz, py::array_t<int> &output) {
  return apply_calendar_op(timestamps, tz, month, output);
}

// hour calendar op
int hour(absl::CivilSecond value) { return value.hour(); }
std::string calendar_hour(const py::array_t<double> &timestamps,
                          const py::object tz, py::array_t<int> &output) {
  return apply_calendar_op(timestamps, tz, hour, output);
}

// day of month calendar op
int day_of_month(absl::CivilSecond value) { return value.day(); }

std::string calendar_day_of_month(const py::array_t<double> &timestamps,
                                  const py::object tz,
                                  py::array_t<int> &output) {
  return apply_calendar_op(timestamps, tz, day_of_month, output);
}

// minute calendar op
int minute(absl::CivilSecond value) { return value.minute(); }

std::string calendar_minute(const py::array_t<double> &timestamps,
                            const py::object tz, py::array_t<int> &output) {
  return apply_calendar_op(timestamps, tz, minute, output);
}

// second calendar op
int second(absl::CivilSecond value) { return value.second(); }

std::string calendar_second(const py::array_t<double> &timestamps,
                            const py::object tz, py::array_t<int> &output) {
  return apply_calendar_op(timestamps, tz, second, output);
}

// day of year calendar op
int day_of_year(absl::CivilSecond value) { return absl::GetYearDay(value); }

std::string calendar_day_of_year(const py::array_t<double> &timestamps,
                                 const py::object tz,
                                 py::array_t<int> &output) {
  return apply_calendar_op(timestamps, tz, day_of_year, output);
}

// day of week calendar op
int day_of_week(absl::CivilSecond value) {
  return map_week_day(absl::GetWeekday(value));
}

std::string calendar_day_of_week(const py::array_t<double> &timestamps,
                                 const py::object tz,
                                 py::array_t<int> &output) {
  return apply_calendar_op(timestamps, tz, day_of_week, output);
}

// isoweek
int iso_week(absl::CivilSecond value) {
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
    if ((iso_week > -2) or (iso_week == -2 and is_leapyear(year - 1))) {
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
}

std::string calendar_isoweek(const py::array_t<double> &timestamps,
                             const py::object tz, py::array_t<int> &output) {
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