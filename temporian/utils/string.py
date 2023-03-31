"""String utilities."""


def indent(text: str, num_spaces: int = 4) -> str:
    """Indents a string."""

    block = " " * num_spaces
    return block + block.join(text.splitlines(True))
