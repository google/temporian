#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "absl/time/time.h"

#include "temporian/implementation/numpy_cc/operators/common.h"

namespace {
namespace py = pybind11;

static absl::TimeZone parse_tz(const py::object tz) {
  if (py::isinstance<py::int_>(tz)) {
    // handle tz as an int
    const int int_tz = tz.cast<int>();
    // tz is expresed in hours, needs to be converted to seconds
    return absl::FixedTimeZone(int_tz * 60 * 60);

  } else if (py::isinstance<py::float_>(tz)) {
    // handle tz as an float
    const float float_tz = tz.cast<float>() * 60 * 60;
    const int int_tz = static_cast<int>(std::round(float_tz));
    // tz is expresed in hours, needs to be converted to seconds
    return absl::FixedTimeZone(int_tz);

  } else if (py::isinstance<py::str>(tz)) {
    // handle tz as an string
    const std::string str_tz = tz.cast<std::string>();
    absl::TimeZone parsed_tz;
    if (!absl::LoadTimeZone(str_tz, &parsed_tz)) {
      throw std::invalid_argument("Invalid timezone '" + str_tz +
                                  "'. Only names defined in the IANA timezone" +
                                  "database are valid");
    }
    return parsed_tz;

  } else {
    throw std::invalid_argument("Unsupported argument type for \"tz\" argument");
  }
}

// TODO: support week starting on Sunday
static int map_week_day(const absl::Weekday &wd) {
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
  std::invalid_argument("Invalid weekday");
}

// Template for iterating a timestamps array, converting to civil time
// and apply a calendar_op to each one
template <typename OUTPUT, typename TCALENDAR_OP>
py::array_t<OUTPUT> apply_calendar_op(const py::array_t<double> &timestamps,
                               const py::object tz) {
  TCALENDAR_OP calendar_op;

  auto parsed_tz = parse_tz(tz);

  // Input size
  const Idx n_events = timestamps.shape(0);

  // Allocate output array
  auto output = py::array_t<int>(n_events);

  // Access raw input / output data
  auto v_output = output.mutable_unchecked<1>();
  auto v_timestamps = timestamps.unchecked<1>();

  for (Idx i = 0; i < n_events; i++) {
    // Create absolute time in UTC
    auto nanos = static_cast<int64_t>(v_timestamps[i] * 1e9);
    auto utc_time = absl::FromUnixNanos(nanos);

    // Convert to civil time and get the isoweek
    auto local_time = absl::ToCivilSecond(utc_time, parsed_tz);
    v_output[i] = calendar_op.Get(local_time);
  }

  return output;
}

// Calendar operations interface
template <typename OUTPUT> struct CALENDAR_OP {
  virtual ~CALENDAR_OP() = default;
  virtual OUTPUT Get(absl::CivilSecond value) = 0;
};

// year calendar op
struct Year : CALENDAR_OP<int> {
  int Get(absl::CivilSecond value) override {
    return value.year();
  }
};

py::array_t<int> calendar_year(const py::array_t<double> &timestamps,
                               const py::object tz) {
  return apply_calendar_op<int, Year>(timestamps, tz);
}

// month calendar op
struct Month : CALENDAR_OP<int> {
  int Get(absl::CivilSecond value) override {
    return value.month();
  }
};

py::array_t<int> calendar_month(const py::array_t<double> &timestamps,
                               const py::object tz) {
  return apply_calendar_op<int, Month>(timestamps, tz);
}

// hour calendar op
struct Hour : CALENDAR_OP<int> {
  int Get(absl::CivilSecond value) override {
    return value.hour();
  }
};
py::array_t<int> calendar_hour(const py::array_t<double> &timestamps,
                               const py::object tz) {
  return apply_calendar_op<int, Hour>(timestamps, tz);
}

// day of month calendar op
struct DayOfMonth : CALENDAR_OP<int> {
  int Get(absl::CivilSecond value) override {
    return value.day();
  }
};

py::array_t<int> calendar_day_of_month(const py::array_t<double> &timestamps,
                               const py::object tz) {
  return apply_calendar_op<int, DayOfMonth>(timestamps, tz);
}

// minute calendar op
struct Minute : CALENDAR_OP<int> {
  int Get(absl::CivilSecond value) override {
    return value.minute();
  }
};

py::array_t<int> calendar_minute(const py::array_t<double> &timestamps,
                               const py::object tz) {
  return apply_calendar_op<int, Minute>(timestamps, tz);
}

// second calendar op
struct Second : CALENDAR_OP<int> {
  int Get(absl::CivilSecond value) override {
    return value.second();
  }
};

py::array_t<int> calendar_second(const py::array_t<double> &timestamps,
                               const py::object tz) {
  return apply_calendar_op<int, Second>(timestamps, tz);
}

// day of year calendar op
struct DayOfYear : CALENDAR_OP<int> {
  int Get(absl::CivilSecond value) override {
    return absl::GetYearDay(value);
  }
};

py::array_t<int> calendar_day_of_year(const py::array_t<double> &timestamps,
                               const py::object tz) {
  return apply_calendar_op<int, DayOfYear>(timestamps, tz);
}

// day of week calendar op
struct DayOfWeek : CALENDAR_OP<int> {
  int Get(absl::CivilSecond value) override {
    return map_week_day(absl::GetWeekday(value));
  }
};

py::array_t<int> calendar_day_of_week(const py::array_t<double> &timestamps,
                               const py::object tz) {
  return apply_calendar_op<int, DayOfWeek>(timestamps, tz);
}


// isoweek

struct IsoWeek : CALENDAR_OP<int> {
  // directly translated to cpp from panda's ccalendar.pyx implementation
  // https://github.com/pandas-dev/pandas/blob/1c606d5f014c5296d6028af28001311b67ee3721/pandas/_libs/tslibs/ccalendar.pyx
  int is_leapyear(int64_t year) {
    return ((year & 0x3) == 0 and  // year % 4 == 0
            ((year % 100) != 0 or (year % 400) == 0));
  }

  int Get(absl::CivilSecond value) override {
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
};

py::array_t<int> calendar_isoweek(const py::array_t<double> &timestamps,
                               const py::object tz) {
  return apply_calendar_op<int, IsoWeek>(timestamps, tz);
}

}  // namespace

void init_calendar_ops(py::module &m) {
  m.def("calendar_year", &calendar_year, "", py::arg("timestamps").noconvert(),
        py::arg("tz"));
  m.def("calendar_month", &calendar_month, "", py::arg("timestamps").noconvert(),
        py::arg("tz"));
  m.def("calendar_day_of_month", &calendar_day_of_month, "", py::arg("timestamps").noconvert(),
        py::arg("tz"));
  m.def("calendar_hour", &calendar_hour, "", py::arg("timestamps").noconvert(),
        py::arg("tz"));
  m.def("calendar_minute", &calendar_minute, "", py::arg("timestamps").noconvert(),
        py::arg("tz"));
  m.def("calendar_second", &calendar_second, "", py::arg("timestamps").noconvert(),
        py::arg("tz"));
  m.def("calendar_day_of_year", &calendar_day_of_year, "", py::arg("timestamps").noconvert(),
        py::arg("tz"));
  m.def("calendar_day_of_week", &calendar_day_of_week, "", py::arg("timestamps").noconvert(),
        py::arg("tz"));
  m.def("calendar_isoweek", &calendar_isoweek, "", py::arg("timestamps").noconvert(),
        py::arg("tz"));
}