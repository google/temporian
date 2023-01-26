from temporal_feature_processor.implementation.pandas import evaluator as pandas_evaluator
from temporal_feature_processor.implementation.pandas.data import event as pandas_event


def raise_(exception: Exception):
  raise exception


BACKENDS = {
    "cpp": {
        "read_csv_fn":
            lambda path: raise_(NotImplementedError),
        "evaluate_schedule_fn":
            lambda data, schedule: raise_(NotImplementedError),
    },
    "pandas": {
        "read_csv_fn": pandas_event.pandas_event_from_csv,
        "evaluate_schedule_fn": pandas_evaluator.evaluate_schedule
    }
}
