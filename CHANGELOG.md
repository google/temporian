# Release notes

## Latest changes (unreleased)

### Features

- Add `filter_moving_count` operator.

### Improvements

### Fixes

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
