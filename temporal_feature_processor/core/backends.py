from temporal_feature_processor.implementation.pandas import evaluator as pandas_evaluator


def raise_(exception: Exception):
  raise exception


BACKENDS = {
    "cpp": lambda data, schedule: raise_(NotImplementedError),
    "pandas": pandas_evaluator.evaluate_schedule
}
