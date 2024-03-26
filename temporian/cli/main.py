"""CLI to operate on event-set.

Support:
    inspect: Print the event set in the terminal.
    view: Plot the event-set interactively.

Usage example:

    # In the project
    bazel run --config=linux //temporian/cli:main -- view data.csv

    # With pip
    python -m temporian view data.csv

    # Inspect
    python -m temporian inspect data.csv

    # View
    python -m temporian view data.csv

    # View multiple files
    python -m temporian view data_1.csv data_2.csv

    # Load TensorFlow Record files
    python -m temporian view data.tfr.gz:tfr:schema.pbtxt

    # Change the display index
    python -m temporian view data.csv --index=A --index=B

    # Display global help
    python -m temporian --help

    # Display help about specific command
    python -m temporian view --help

Input event sets:
    Event set paths support three formats:
        <path>: Path to a csv file
        <path>:<format>: Path to a file with a specific format. Format can be
            "csv" or "tfr".
        <path>:<format>:<schema path>: Path with custom schema serialized to
            file e.g. `evtset.schema.to_proto_file(<schema path>)`.

"""

from typing import Any, List

import argparse
import temporian as tp


FlatArgs = Any


def add_load_evtset_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments about input eventset paths to a parser."""
    parser.add_argument("path", nargs="+")
    parser.add_argument("--index", nargs="*")


def parse_arguments() -> FlatArgs:
    """Parse the cli user arguments."""
    parser = argparse.ArgumentParser(
        description="CLI to interract with event-sets."
    )
    subparsers = parser.add_subparsers(title="mode", required=True, dest="mode")

    parser_view = subparsers.add_parser(
        "view",
        help=(
            "Plot the event-set interactively. Controls: left mouse:"
            " query. right mouse: navigation. mouse wheel: time zoom. mouse"
            " wheel + right click: feature zoom."
        ),
    )
    add_load_evtset_args(parser_view)
    parser_view.add_argument("--font_size", default=14, type=int)

    inspect_view = subparsers.add_parser(
        "inspect", help="Print the event set in the terminal."
    )
    add_load_evtset_args(inspect_view)

    args = parser.parse_args()
    print(f"Arguments: {args}")
    return args


def load_evtsets(args: FlatArgs) -> List[tp.EventSet]:
    """Loads eventsets."""

    evtsets = []
    for path in args.path:
        parts = path.split(":")

        if len(parts) == 1:
            evtset_path, format, schema_path = parts[0], "csv", None
        elif len(parts) == 2:
            evtset_path, format, schema_path = parts[0], parts[1], None
        elif len(parts) == 3:
            evtset_path, format, schema_path = parts[0], parts[1], parts[2]
        else:
            raise ValueError(
                f"Unexpected path format {path!r}."
                " Expecting <path>, <path>:<format>, or"
                " <path>:<format>:<schema path>."
            )

        if format == "csv":
            if schema_path is not None:
                raise ValueError("format csv does not needs a schema")
            print(f"Loading CSV file: {evtset_path}")
            evtset = tp.from_csv(evtset_path)
        elif format == "tfr":
            if schema_path is None:
                raise ValueError("format tfr needs a schema")
            print(f"Loading schema file: {schema_path}")
            schema = tp.Schema.from_proto_file(schema_path)
            print(f"Loading TFR file: {evtset_path}")
            evtset = tp.from_tensorflow_record(
                evtset_path, schema=schema, num_parallel_reads=20
            )
        else:
            raise ValueError(f"Unknown format {format}")

        if args.index:
            evtset = evtset.set_index(args.index)

        evtsets.append(evtset)

    return evtsets


def inspect(args):
    """Inspect action."""
    for evtset in load_evtsets(args):
        print(evtset)
        print("====================")


def view(args):
    """View action."""
    from temporian.cli import view

    view.main(args, load_evtsets(args))


def main():
    args = parse_arguments()

    if args.mode == "view":
        view(args)
    elif args.mode == "inspect":
        inspect(args)
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()
