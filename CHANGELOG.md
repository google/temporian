# Changelog

## HEAD (might become 0.1.3)

This is the first operational version of Temporian for users. The list whole and
detailed list of features is too long to be listed. The top features are:

### Feature

- Pypi release.
- 72 operators.
- Execution in eager, compiled mode, and graph mode.
- IO Support for Pandas, CSV, Numpy and TensorFlow datasets.
- Static and interactive plotting.
- Documentation (3 minutes intro, user guide and API references).
- 5 tutorials.

### Fix

- Proper error message when using distributed training on more than 2^31
  (i.e., ~2B) examples while compiling YDF with 32-bits example index.
