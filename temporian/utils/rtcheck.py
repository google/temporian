"""Runtime checking."""

from __future__ import annotations

from functools import wraps
import logging


from typing import List, Set, Dict, Optional, Union, Tuple, Any
import inspect
import typing

from temporian.implementation.numpy.data.event_set import EventSet
from temporian.core.data.node import EventSetNode
from temporian.core.event_set_ops import EventSetOperations

# Number of elements to check in a structure e.g. a list.
_NUM_CHECK_STRUCT = 3

# If true, print details during runtime checking.
_DEBUG = False

# Move to "config".
_ERROR_RAISES_EXCEPTION = True


def runtime_check_raise_exception(value: bool) -> None:
    """Sets the behavior when RT Check finds an issue.

    Arg:
        value: If True, a RT check error will raise an exception. If False, a RT
          check error will print a warning.
    """

    global _ERROR_RAISES_EXCEPTION
    _ERROR_RAISES_EXCEPTION = value


class _Trace:
    """A trace collects a description of the unfolding of a value.

    Usage example:

        a = _Trace()
        b = a.add_context("something")
        c = a.add_context("something else")
        c.exception("Something wrong")

    """

    def __init__(self):
        self.context = []

    def exception(self, value: str):
        """Raises an exception with the current context and the value."""

        disable_error_info = (
            "Note: You can disable runtime checking with"
            " tp.runtime_check_raise_exception(False)."
        )

        raise ValueError(
            " ".join([value, "\n"] + self.context + ["\n", disable_error_info])
        )

    def add_context(self, value: str) -> _Trace:
        """Creates a new trace with the current's trace context and value."""

        n = _Trace()
        n.context.extend(self.context)
        n.context.append(value)
        return n


def _base_error(value, annotation):
    """Text of a type mismatch error."""

    return (
        f"Expecting value of type {annotation} but received value of type"
        f' {type(value)}. The value is "{value}".'
    )


def _check_annotation(trace: _Trace, is_compiled: bool, value, annotation):
    """Checks recursively that "value" is compatible with "annotation"."""

    if _DEBUG:
        logging.info(
            "Checking %s (%s) against %s (%s)",
            value,
            type(value),
            annotation,
            type(annotation),
        )

    if isinstance(annotation, str):
        # Unfold annotation
        try:
            annotation = typing.get_type_hints(annotation)
        except ValueError:
            logging.warning("Cannot unfold annotation %s", annotation)

    if annotation in (
        EventSet,
        EventSetNode,
        EventSetOperations,
    ) and isinstance(value, (EventSet, EventSetNode, EventSetOperations)):
        return

    if annotation in [inspect._empty, Any, Optional]:
        # No annotation information
        return None

    annotation_args = typing.get_args(annotation)

    if isinstance(annotation, typing._GenericAlias):
        # The annotation is a possibly composed type e.g. List, List[int]

        origin = typing.get_origin(annotation)
        assert origin is not None

        if origin is not Union:
            if not isinstance(value, origin):
                # The origin (e.g. "list" in "List[int]") is wrong.
                trace.exception(_base_error(value, annotation))

        # Check the sub-argument in composed types.

        if origin in [List, Set, list, set]:
            _check_annotation_list_or_set_or_uniform_tuple(
                trace, is_compiled, value, annotation_args
            )
        elif origin in [dict, Dict]:
            _check_annotation_dict(trace, is_compiled, value, annotation_args)
        elif origin is Union:
            _check_annotation_union(trace, is_compiled, value, annotation_args)
        elif origin in [tuple, Tuple]:
            _check_annotation_tuple(trace, is_compiled, value, annotation_args)
        else:
            if _DEBUG:
                logging.warning(
                    "Unknown generic alias annotation %s (%s) with origin=%s",
                    annotation,
                    type(annotation),
                    origin,
                )
    else:
        try:
            is_instance_result = isinstance(value, annotation)
        except TypeError:
            if _DEBUG:
                logging.warning(
                    "Cannot check %s (%s) against %s (%s)",
                    value,
                    type(value),
                    annotation,
                    type(annotation),
                )
            return None

        if not is_instance_result:
            trace.exception(_base_error(value, annotation))

    return None


def _check_annotation_list_or_set_or_uniform_tuple(
    trace: _Trace, is_compiled: bool, value, annotation_args
):
    if len(annotation_args) == 0:
        pass  # No sub-type to check
    elif len(annotation_args) == 1:
        num_checks = min(len(value), _NUM_CHECK_STRUCT)
        idx = 0
        for v in value:
            if idx >= num_checks:
                break
            idx += 1

            _check_annotation(
                trace.add_context(
                    "When checking the content of a list, set or tuple."
                ),
                is_compiled,
                v,
                annotation_args[0],
            )
    else:
        trace.exception(
            "List and set annotations require zero or one arguments."
        )


def _check_annotation_union(
    trace: _Trace, is_compiled: bool, value, annotation_args
):
    for arg in annotation_args:
        try:
            _check_annotation(
                trace.add_context("When checking the content of a union."),
                is_compiled,
                value,
                arg,
            )
            return
        except ValueError:
            pass

    trace.exception(
        f'Non matching type for "{type(value)}" in the union {annotation_args}.'
    )


def _check_annotation_tuple(
    trace: _Trace, is_compiled: bool, value, annotation_args
):
    if len(annotation_args) != len(value):
        trace.add_context("Wrong number of items in tuple.")

    for val, arg in zip(value, annotation_args):
        _check_annotation(
            trace.add_context("When checking the content of a tuple."),
            is_compiled,
            val,
            arg,
        )


def _check_annotation_dict(
    trace: _Trace, is_compiled: bool, value, annotation_args
):
    if len(annotation_args) == 0:
        pass  # No sub-type to check
    elif len(annotation_args) == 2:
        num_checks = min(len(value), _NUM_CHECK_STRUCT)
        idx = 0
        for k, v in value.items():
            if idx >= num_checks:
                break
            idx += 1

            _check_annotation(
                trace.add_context("When checking a dict key."),
                is_compiled,
                k,
                annotation_args[0],
            )
            _check_annotation(
                trace.add_context("When checking a dict value."),
                is_compiled,
                v,
                annotation_args[1],
            )

    else:
        trace.exception("Dict annotation require zero or two arguments.")


def rtcheck(fn):
    """Annotation that check the arguments and outputs of a function at runtime.

    @rtcheck checks, at runtime, that the type hints of the arguments and output
    of a function are satisfied.

    Usage example:
        ```python
        @rtcheck
        def f(a, b: int, c: str = "aze") -> List[str]:
            return ["hello", "world"]

        f(1, 2, "a") # Ok
        f(1, 2, 3) # Fails
        ```

    If combined with @compile, @rtcheck should be applied after @compile (i.e.
    place @compile just below @rtcheck in the code).

    This code only support what is required by Temporian API.

    Does not support typing.GenericTypeAlias e.g. list[int]. Use List[int]
    instead.

    Args:
        fn: Function to instrument.

    Returns:
        Instrumented function.
    """
    is_compiled = False

    if hasattr(fn, "__wrapped__"):
        signature_fn = fn.__wrapped__
        if hasattr(signature_fn, "is_tp_compiled"):
            is_compiled = getattr(signature_fn, "is_tp_compiled")
    else:
        signature_fn = fn
    signature = inspect.signature(signature_fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            # Check inputs
            all_args = signature.bind(*args, **kwargs)
            for arg_key, arg_value in all_args.arguments.items():
                trace = _Trace().add_context(
                    f'When checking the argument "{arg_key}" of function'
                    f' "{fn.__name__}".'
                )

                if arg_key not in signature.parameters:
                    raise ValueError(f'Unexpected argument "{arg_key}"')
                param = signature.parameters[arg_key]

                if param.kind in [
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ]:
                    _check_annotation(
                        trace,
                        is_compiled,
                        arg_value,
                        param.annotation,
                    )
                elif param.kind is inspect.Parameter.VAR_POSITIONAL:
                    _check_annotation_list_or_set_or_uniform_tuple(
                        trace,
                        is_compiled,
                        arg_value,
                        [param.annotation],
                    )
                elif param.kind is inspect.Parameter.VAR_KEYWORD:
                    for sub_key, sub_value in arg_value.items():
                        _check_annotation(
                            _Trace().add_context(
                                f'When checking the key "{sub_key}" of argument'
                                f' "{arg_key}" of function "{fn.__name__}".'
                            ),
                            is_compiled,
                            sub_value,
                            param.annotation,
                        )
        except ValueError as e:
            if _ERROR_RAISES_EXCEPTION:
                # Reset the stack trace of the exception.
                e.__traceback__ = None
                raise e
            else:
                logging.warning("%s", str(e))

        output = fn(*args, **kwargs)

        try:
            # Check outputs
            trace = _Trace().add_context(
                f'When checking the returned value of function "{fn.__name__}".'
            )
            _check_annotation(
                trace,
                is_compiled,
                output,
                signature.return_annotation,
            )
        except ValueError as e:
            if _ERROR_RAISES_EXCEPTION:
                # Reset the stack trace of the exception.
                e.__traceback__ = None
                raise e
            else:
                logging.warning("%s", str(e))

        return output

    setattr(wrapper, "_rtcheck", True)
    return wrapper
