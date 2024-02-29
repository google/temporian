#include <pybind11/pybind11.h>

#include <cstdint>
#include <optional>

#include "absl/time/time.h"

struct MyTime {
  int64_t seconds_since_epoch;
  int wday;  // 0-6, where 0 is Sunday.
};

// Tests if a year is a leap year.
bool IsLeapYear(int year);

// Gets the UTC Unix timestamps and week-day of a UTC date.
// If the date is invalid (e.g., day=29 for a non-leap year, month>12) returns
// nothing.
std::optional<MyTime> UTCMkTime(const int year, const int month, const int day,
                                const int hour, const int minute,
                                const int second);

typedef std::optional<std::string> ErrorString;
namespace py = pybind11;

ErrorString parse_tz(const py::object &tz, absl::TimeZone &parsed_tz);
int map_week_day(const absl::Weekday &wd);