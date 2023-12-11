#include "absl/time/time.h"

static absl::TimeZone handle_tz(const pybind11::object tz) {
  if (pybind11::isinstance<pybind11::int_>(tz)) {
    // handle tz as an int
    int int_tz = tz.cast<int>();
    // tz is expresed in hours, needs to be converted to seconds
    return absl::FixedTimeZone(int_tz * 60 * 60);

  } else if (pybind11::isinstance<pybind11::float_>(tz)) {
    // handle tz as an float
    float float_tz = tz.cast<float>() * 60 * 60;
    int int_tz = static_cast<int>(std::round(float_tz));
    // tz is expresed in hours, needs to be converted to seconds
    return absl::FixedTimeZone(int_tz);

  } else if (pybind11::isinstance<pybind11::str>(tz)) {
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