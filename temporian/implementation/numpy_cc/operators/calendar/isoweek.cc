#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <ctime>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "temporian/implementation/numpy_cc/operators/calendar/common.h"
#include "temporian/implementation/numpy_cc/operators/common.h"

namespace {
namespace py = pybind11;

// directly translated to cpp from panda's ccalendar.pyx implementation
int is_leapyear(int64_t year) {
  return ((year & 0x3) == 0 and  // year % 4 == 0
          ((year % 100) != 0 or (year % 400) == 0));
}

int isoweek(const absl::CivilDay &cd) {
  auto doy = absl::GetYearDay(cd);
  auto dow = map_week_day(absl::GetWeekday(cd));
  auto day = cd.day();
  auto year = cd.year();
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

py::array_t<int> calendar_isoweek(const py::array_t<double> &timestamps,
                                  const py::object tz) {
  auto parsed_tz = handle_tz(tz);

  // Input size
  const Idx n_events = timestamps.shape(0);

  // Allocate output array
  auto weeks = py::array_t<int>(n_events);

  // Access raw input / output data
  auto v_weeks = weeks.mutable_unchecked<1>();
  auto v_timestamps = timestamps.unchecked<1>();

  for (Idx i = 0; i < n_events; i++) {
    // Create absolute time in UTC
    auto nanos = static_cast<int64_t>(v_timestamps[i] * 1e9);
    auto utc_time = absl::FromUnixNanos(nanos);

    // Convert to civil time and get the isoweek
    auto local_time = absl::ToCivilDay(utc_time, parsed_tz);
    v_weeks[i] = isoweek(local_time);
  }

  return weeks;
}
}  // namespace

void init_calendar_isoweek(py::module &m) {
  m.def("calendar_isoweek", &calendar_isoweek, "",
        py::arg("timestamps").noconvert(), py::arg("tz"));
}