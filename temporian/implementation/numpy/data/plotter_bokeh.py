"""Bokeh plotting backend."""

from typing import Optional, List, Any, Set

from bokeh.plotting import figure
from bokeh.models import Range1d
from bokeh.io import output_notebook, show
from bokeh.layouts import gridplot
from bokeh.models import CustomJS
from bokeh.palettes import Dark2_5 as colors
import numpy as np

from temporian.core.data.duration_utils import (
    convert_timestamp_to_datetime,
)
from temporian.implementation.numpy.data.plotter_base import (
    Options,
    Style,
    PlotterBackend,
)


class Plotter(PlotterBackend):
    output_backend = "canvas"

    def __init__(self, num_plots: int, options: Options):
        super().__init__(num_plots, options)

        self.figs = []
        self.options = options

    def new_subplot(
        self,
        title: Optional[str],
        num_items: int,
        is_unix_timestamp: bool,
    ):
        # We need to know if the data is categorical before creating the figure.
        self.cur_fig = None
        self.cur_title = title
        self.cur_is_unix_timestamp = is_unix_timestamp

    def finalize_subplot(
        self,
    ):
        if self.cur_fig is not None:
            self.figs.append(self.cur_fig)

    def ensure_cur_is_available(self, categorical_values: Optional[List[str]]):
        if self.cur_fig is None:
            self.cur_fig = self.create_fig(
                self.cur_title,
                self.cur_is_unix_timestamp,
                self.options,
                categorical_values,
            )
            self.has_categorical_values = categorical_values is not None
        else:
            if self.has_categorical_values != (categorical_values is not None):
                raise ValueError(
                    "Cannot plot string and non-string features in the same"
                    " sub-plot."
                )

    def plot_feature(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        name: Optional[str],
        style: Style,
        color_idx: int,
    ):
        data = {"x": xs, "y": ys, "color": colors[color_idx % len(colors)]}
        is_string = ys.dtype.type is np.str_ or ys.dtype.type is np.bytes_

        if is_string:
            ys = ys.astype(np.str_)
            unique_ys_values = sorted(list(set(ys)))
            self.ensure_cur_is_available(unique_ys_values)

            self.cur_fig.circle(x=xs, y=ys)
        elif style == Style.line:
            self.ensure_cur_is_available(None)
            self.cur_fig.line(**data)
        elif style == Style.marker:
            self.ensure_cur_is_available(None)
            self.cur_fig.scatter(**data)
        elif style == Style.vline:
            self.ensure_cur_is_available(None)
            self.cur_fig.scatter(**data)
        else:
            raise ValueError("Non implemented style")

    def plot_sampling(
        self,
        xs: np.ndarray,
        color_idx: int,
        name: str,
    ):
        self.ensure_cur_is_available(None)

        # TODO: Use "name"
        color = colors[color_idx % len(colors)]
        data = {"x": xs, "y": np.zeros(len(xs)), "color": color}
        self.cur_fig.scatter(**data)

    def finalize(self):
        if len(self.figs) > 1:
            # Sync x-axes
            js_vars = [
                f"p{fig_idx+1}_x_range" for fig_idx, fig in enumerate(self.figs)
            ]
            js_inputs = {}
            core_code = ""
            for js_var, fig in zip(js_vars, self.figs):
                js_inputs[js_var] = fig.x_range

                sub_core_code = "\n".join(
                    [
                        f"""
                {other_js_var}.start = start;
                {other_js_var}.end = end;
                """
                        for other_js_var in js_vars
                        if other_js_var != js_var
                    ]
                )

                core_code += f"""
                if (cb_obj == {js_var}) {{
                    const start =  {js_var}.start;
                    const end =  {js_var}.end;
                    {sub_core_code}
                }}
                """

            callback = CustomJS(args=js_inputs, code=core_code)

            for fig in self.figs:
                fig.x_range.js_on_change("start", callback)
                fig.x_range.js_on_change("end", callback)

            figure_set = gridplot(
                [[f] for f in self.figs],
                merge_tools=True,
                toolbar_location="right",
                toolbar_options=dict(logo=None),
            )
        else:
            figure_set = self.figs[0]
            figure_set.toolbar.logo = None

        output_notebook(hide_banner=True)
        show(figure_set)
        return figure_set

    def create_fig(
        self,
        title: Optional[str],
        is_unix_timestamp: bool,
        options: Options,
        categorical_values: Optional[List[str]],
    ):
        tools = [
            "xpan",
            "pan",
            "xwheel_zoom",
            "ywheel_zoom",
            "box_zoom",
            "reset",
            "undo",
            "save",
            "hover",
        ]

        fig_args = {}
        if is_unix_timestamp:
            fig_args["x_axis_type"] = "datetime"
        if title:
            fig_args["title"] = title

        if options.min_time is not None or options.max_time is not None:
            args = {}
            if options.min_time is not None:
                args["start"] = (
                    convert_timestamp_to_datetime(options.min_time)
                    if is_unix_timestamp
                    else options.min_time
                )
            if options.max_time is not None:
                args["end"] = (
                    convert_timestamp_to_datetime(options.max_time)
                    if is_unix_timestamp
                    else options.max_time
                )
            fig_args["x_range"] = Range1d(**args)

        if categorical_values is not None:
            fig_args["y_range"] = categorical_values

        # Note: Ranges cannot be set after the figure is created see:
        # https://discourse.bokeh.org/t/updating-y-range-to-categorical/2397/3
        fig = figure(
            width=options.width_px,
            height=options.height_per_plot_px,
            tools=tools,
            output_backend=self.output_backend,
            **fig_args,
        )

        return fig


class PlotterWebGL(Plotter):
    output_backend = "webgl"
