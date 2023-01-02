from temporal_feature_processor.sequences import EventSequence, FeatureSequence

from .base import Operator


class AssignOperator(Operator):
    def __call__(self, event: EventSequence, feature: FeatureSequence) -> EventSequence:
        """Assign a feature to an event.
        Input event and feature must have same index.
        Feature cannot have more than one row for a single index + timestamp occurence.
        Output event will have same exact index and timestamps as input one.
        Assignment can be understood as a left join on the index and timestamp columns.

        Args:
            event (EventSequence): event to assign the feature to.
            feature (FeatureSequence): feature to assign to the event.

        Returns:
            EventSequence: a new event with the feature assigned.
        """
        # TODO: implement logic.
        pass
