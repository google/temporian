"""Reads csv files in Beam."""

import apache_beam as beam


class _Run(beam.PTransform):
    def __init__(self, inputs, outputs):
        pass

    def expand(self, pipe):
        return pipe


run = _Run
