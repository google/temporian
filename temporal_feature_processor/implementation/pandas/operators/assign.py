from temporal_feature_processor.implementation.pandas.data.event import PandasEvent

from .base import PandasOperator


class PandasAssignOperator(PandasOperator):

  def __call__(self, event: PandasEvent, features: PandasEvent) -> PandasEvent:
    """Assign features to an event.
        Input event and features must have same index.
        Features cannot have more than one row for a single index + timestamp occurence.
        Output event will have same exact index and timestamps as input one.
        Assignment can be understood as a left join on the index and timestamp columns.

        Args:
            event (PandasEvent): event to assign the feature to.
            feature (PandasEvent): features to assign to the event.

        Returns:
            PandasEvent: a new event with the features assigned.
        """
    # TODO: implement logic.
    pass
