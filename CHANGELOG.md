# Release notes

## Latest changes (unreleased)

### Features

### Improvements

- Added `before_first` and `after_last` parameters to `EventSet.tick` and `EventSet.tick_calendar`

### Fixes

## 0.8.1

Adding missing wheels for macos, no changes to the library

## 0.8.0

### Features

- Add `EventSet.moving_product()` and `EventSet.cumprod()` operators. @akshatvishu
- Add `to.to_numpy()`. @nagavenkateshgavini
- Add trigonometric functions `EventSet.arccos()`, `EventSet.arcsin()`, `EventSet.arctan()`, `EventSet.cos()`, `EventSet.sin()`, and `EventSet.tan()`. @akshatvishu

### Improvements

- Speed up of calendar operations (now implemented in c++)
- Add `force_garbage_collector_interval` to `tp.compile`.
- Improved worst case time complexity from quadratic to linear for moving min/max operator.
- Show first and last events instead of only first ones when displaying an EventSet. @jtaylor205
- Exposed `tp.FeatureSchema` and `tp.IndexSchema`
- `EventSet.propagate` now works even if both arguments don't have the same index values.
- Add `font_scale` param to `tp.plot`
- Speed-up `tp.plot` on evsets with a large number of index values.
- `tp.plot` now works even when evsets have different index values.
- Allow `tp.from_tensorflow_record` to consume a list of paths.
- Add parameters `num_parallel_reads` and `buffer_size` to `tp.from_tensorflow_record`.
- Check that timestamps are sorted on `tp.from_tensorflow_record`

### Fixes

- Fixed a bug with `EventSet.tick_calendar` and daylight savings time.
- Fixed a bug in `EventSet.tick_calendar` where it would miss the first tick under certain conditions
- Fixed a bug in `EventSet.tick` where it would miss the last tick under certain conditions
- Fixed a bug with calendar operations and daylight savings time.

### Thanks

In adition to the contributors mentioned above, thanks to @umbr4g3, @jsoref, and @tanaysd for improvements to the Github Actions and profiling.

## 0.7.0

### Features

- Add `tp.from_parquet()` and `tp.to_parquet()`.
- Add `EventSet.fillna()` operator.

### Improvements

- Add support for pip build on Windows.
- Documentation improvements.
- Add `timestamps` parameter to `tp.from_pandas()`.
- Add implicit casting in `EventSet.where()` operator.
- Add support for list argument in `EventSet.rename()` operator.

## 0.1.6

### Features

- Support for `timezone` argument in all calendar operators.
- Add `drop()` operator to drop features.
- Add `assign()` operator to assign features.
- Add `before()` and `after()` operators.

### Improvements

- Improve error messages for type mismatch in window operators.
- Improve structure of docs site.
- Support exporting timestamps as datetimes in `tp.to_pandas()`.
- Remove inputs limit in `glue()` and `combine()`.

### Fixes

- Use `wday=0` for Mondays in `tick_calendar` (like `calendar_day_of_week`).
- Support bool in `DType.missing_value()`.
- Show `EventSet`'s magic methods in docs.

## 0.1.5

### Features

- Added `EventSet.filter_moving_count()` operator.
- Added `EventSet.map()` operator.
- Added `EventSet.tick_calendar()` operator.
- Added `EventSet.where()` operator.
- Added all moving window operators to Beam execution backend.

### Improvements

- Print `EventSet` timestamps as datetimes instead of float.
- Support `sampling` argument in `EventSet.cumsum()` operator.
- Using utf-8 codec to support non-ascii in string values.
- New `tp.types` module to facilitate access to types used throughout the API.
- Relaxed version requirements for protobuf and pandas.

### Fixes

- Fixed issues when loading timestamps from `np.longlong` and other dtypes.

## 0.1.4

### Features

- Added `EventSet.select_index_values()` operator.
- Added `steps` argument to `EventSet.since_last()` operator.
- Added variable `window_length` option to moving window operators.
- Added unsupervised anomaly detection tutorial.
- Add `until_next` operator.
- Added Beam execution tutorial.
- Added changelog to docs site.

### Improvements

- Added `display_max_feature_dtypes` and `display_max_index_dtypes` options to
  `tp.config`.
- Improved HTML display of an `EventSet`.
- Improvements in Beam execution backend.

### Fixes

- Fixed tutorials opening unreleased versions of the notebooks.

## 0.1.3

This is the first operational version of Temporian for users. The list whole and
detailed list of features is too long to be listed. The main features are:

### Features

- PyPI release.
- 72 operators.
- Execution in eager, compiled mode, and graph mode.
- IO Support for Pandas, CSV, Numpy and TensorFlow datasets.
- Static and interactive plotting.
- Documentation (3 minutes intro, user guide and API references).
- 5 tutorials.
