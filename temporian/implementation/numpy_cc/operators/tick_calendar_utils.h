#include <cstdint>
#include <optional>

struct MyTime {
  int64_t seconds_since_epoch;
  int wday;
};

bool IsLeapYear(int year);

std::optional<MyTime> UTCMkTime(const int year, const int month, const int day,
                                const int hour, const int minute,
                                const int second);