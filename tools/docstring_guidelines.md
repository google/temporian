# Docstring guidelines

Guidelines for writing docstrings for temporian's public API.

Note that these guidelines are useful for internal docstrings too (a.k.a. the
ones intended as documentation for other developers of the library rather than
its users) but we do not enforce it on those. Internal docstrings can be leaner.

## Objective

The only purpose of reference material in general is to describe, as succinctly
as possible, and in a consistent and orderly way. Good technical reference is
essential to provide users with the confidence to do their work.

Users hardly _read_ reference material; they _consult_ it. A docstring's
objective is not to delight its readers with vocabulary and style.

## General guidelines

Docstrings should be:

- Austere.
- Accurate.
- To the point.
- Unambiguous.
- Reliable.
- Consistent (in language, structure, terminology and tone).

Docstrings should do nothing but describe.

Docstrings should not attempt to show how to perform tasks, but may include a
succint example of the correct way to use what its documenting.

## Style

Temporian uses [Google's docstring style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

Note that the descriptions below are a distilled and opinionated version of that
document.

## Concrete style guidelines

### General

Use three-double-quote format `"""`.

A docstring should be composed of:

1. A summary line (one physical line not exceeding 80 chars) terminated by a
   period.
2. A blank line.
3. When writing more (recommended) the rest of the docstring.

Every file should contain license boilerplate.

Docstrings that do not provide any new information should be avoided:

```
def lag_operator_test(self):
    """Test for lag operator."""
```

### Modules

A module's docstring should describe the contents and usage of the module.

Note that in Python every file is a module.

In temporian we do not (yet) consider module docstrings to be mandatory.

Example:

````
    """A one-line summary of the module, terminated by a period.

    Leave one blank line.  The rest of this docstring should contain an
    overall description of the module.  Optionally, it may also contain a brief
    description of exported classes and functions and/or usage examples.

    Example:
        ```python
        foo = ClassFoo()
        bar = foo.FunctionBar()
        ```
    """
````

Note the indentation after the "Example:" clause, which makes mkdocs render it
as a dropdown component, and the use of "```", which makes it be rendered as a
code block. The same applies for class docstrings.

#### Test modules

Module-level docstrings for test files are not required and should only be
included when there is additional information that can be provided.

### Functions and methods

A docstring is **mandatory** for every function that has one or more of the
following properties:

- being part of the public API
- non-trivial size
- non-obvious logic

The docstring's style must be descriptive rather than imperative, i.e.,
`"""Shifts the sampling backwards"""` instead of `"""Shift the sampling
backwards"""`.

The docstring's style must not be prefixed with a noun, i.e. `"""Shifts the
sampling backwards"""` instead of `"""Function that shifts the sampling
backwards"""` or `"""Operator that shifts the sampling backwards"""`, since it
provides no extra information to the reader, who already knows it is a function.

A method that overrides a method from a base class may have a docstring such
as `"""See base class."""`, unless its behavior is substantially different.

Certain aspects of a function should be documented in the special sections
listed below. Each section has a heading, which ends with a colon, and its
content is indented 4 spaces in (consistent with project's indentation).

**Args:**

List each parameter by name.
Do not document the parameter's expected type (type hints are enough).
A description should follow the name, and be separated by a colon followed by a
space.
If the description goes over the max line length, break into a new line with an
additional indentation of 4 spaces.
If a function accepts `*args` and/or `**kwargs` they should be listed as such.
Omit unnecessary prepositions at the start of each arg's description, such as
`The` or `A`.

**Returns: (or Yields: for generators)**

Describe the semantics of the return value.
Do not document the expected return type (type hints are enough).
Do not include the section for functions that return `None`.
If a tuple is returned, describe it as: `Returns: A tuple (a, b) where a is...`.

**Raises:**

List and describe all exceptions that are relevant to the interface.
Use the same name + colon + space and indent style as in **Args:**.
Do not document exceptions that get raised if the API specified in the docstring
is violated.

Example:

````
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Example:
        ```python
        rows = fetch_rows(handle, keys)
        ```

    Args:
        table_handle: Open `smalltable.Table` instance.
        keys: Sequence of strings representing the key of each table row to
            fetch. String keys will be UTF-8 encoded.
        require_all_keys: If `True`,  only rows with values set for all keys
            will be returned.

    Returns:
        Dict mapping keys to the corresponding table row data fetched. Each row
        is represented as a tuple of strings.

        For example:
            ```
            {b'Serak': ('Rigel VII', 'Preparer'),
            b'Zim': ('Irk', 'Invader'),
            b'Lrrr': ('Omicron Persei 8', 'Emperor')}
            ```

        Returned keys are always bytes. If a key from the keys argument is
        missing from the dictionary, then that row was not found in the
        table (and `require_all_keys` must have been False).

    Raises:
        IOError: An error occurred accessing the smalltable.
    """
````

### Classes

Classes should have a docstring below the class definition describing them.

If the class has public attributes, they can be documented here in an
**Attributes:** section and follow the same formatting as a function's **Args:**
section.

Example:

```
    """Summary of class here.

    Longer class information...
    Longer class information...

    Attributes:
        likes_spam: Boolean indicating if we like SPAM or not.
        eggs: Integer count of the eggs we have laid.
    """
```

## Sources

- [Diátaxis framework for documentation authoring](https://diataxis.fr/reference/)
  (recommended read)
- [Google's Python style guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
