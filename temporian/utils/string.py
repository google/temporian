"""String utilities."""


def indent(text: str, num_spaces: int = 4) -> str:
    """Indents a string."""
    block = " " * num_spaces
    return block + block.join(text.splitlines(True))


def pretty_num_bytes(nbytes: int) -> str:
    """Converts a number of bytes in a human readable form."""

    if nbytes > 5e8:
        return f"{(nbytes / 1e9):.1f} GB"
    elif nbytes > 5e5:
        return f"{(nbytes / 1e6):.1f} MB"
    elif nbytes > 5e2:
        return f"{(nbytes / 1e3):.1f} kB"
    else:
        return f"{nbytes} B"
