# Release notes

## Latest changes (unreleased)

### Features

### Improvements

### Fixes

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
