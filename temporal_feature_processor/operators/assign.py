from temporal_feature_processor.sequences import EventSequence

from .base import Operator


class AssignOperator(Operator):
    def __call__(self, event: EventSequence, features: EventSequence) -> EventSequence:
        """Assign features to an event.
        Input event and features must have same index.
        Features cannot have more than one row for a single index + timestamp occurence.
        Output event will have same exact index and timestamps as input one.
        Assignment can be understood as a left join on the index and timestamp columns.

        Args:
            event (EventSequence): event to assign the feature to.
            feature (EventSequence): features to assign to the event.

        Returns:
            EventSequence: a new event with the features assigned.
        """
        # TODO: implement logic.
        pass
