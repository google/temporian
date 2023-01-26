from temporal_feature_processor.implementation.pandas import evaluator as pandas_evaluator
from temporal_feature_processor.implementation.pandas.data import event as pandas_event


def raise_(exception: Exception):
  raise exception


BACKENDS = {
    "cpp": {
        "event":
            lambda: raise_(NotImplementedError),
        "evaluate_schedule_fn":
            lambda data, schedule: raise_(NotImplementedError),
        "read_csv_fn":
            lambda path: raise_(NotImplementedError)
    },
    "pandas": {
        "event": pandas_event.PandasEvent,
        "evaluate_schedule_fn": pandas_evaluator.evaluate_schedule,
        "read_csv_fn": pandas_event.pandas_event_from_csv
    }
}
