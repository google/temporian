"""Viewer implementation."""

from typing import List, Tuple, Set
import dataclasses
import datetime
import re

import numpy as np
from PIL import Image, ImageTk, ImageDraw

import temporian as tp
from temporian.core import typing

try:
    import tkinter as tk
    import tkinter.ttk as ttk
except ModuleNotFoundError as e:
    raise ValueError(
        """\
Cannot import tkinter.
On Linux, run: sudo apt-get install python3-tk
"""
    ) from e


@dataclasses.dataclass
class DisplayItem:
    # Index of event set to display in `Viewer.evtsets`.
    evtset_idx: int
    # Index of feature to display in `Viewer.evtsets[evtset_idx]`
    # If `feature_idx=-1`, display sampling.
    feature_idx: int
    # Display label
    label: str
    # Height on screen in px, before scalling
    height: int


@dataclasses.dataclass
class Vector2i:
    x: int
    y: int

    def copy(self, other: "Vector2i"):
        self.x = other.x
        self.y = other.y

    def set(self, x: int, y: int):
        self.x = x
        self.y = y

    def tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


@dataclasses.dataclass
class TimeSegment:
    begin: float
    end: float

    def av(self, value: float) -> float:
        return (value - self.begin) / (self.end - self.begin)


@dataclasses.dataclass
class BBox2i:
    x: int
    y: int
    w: int
    h: int

    def set_end_x(self, value: int) -> "BBox2i":
        self.w = value - self.x
        return self

    def rect(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def center(self) -> Vector2i:
        return Vector2i(self.x + self.w // 2, self.y + self.h // 2)

    def contains(self, p: Vector2i) -> bool:
        return (
            p.x >= self.x
            and p.x < self.x + self.w
            and p.y >= self.y
            and p.y < self.y + self.h
        )


class Viewer:
    # Window size in px.
    window_size: Vector2i = Vector2i(800, 600)
    # Position of the mouse in px.
    mouse_position: Vector2i = Vector2i(0, 0)
    # Position of the mouse when button is pressed.
    # Used to compute selection windows.
    last_mouse_position: Vector2i = Vector2i(0, 0)

    # Range of time of the data.
    total_time: TimeSegment
    # Range of time beeing displayed.
    display_time: TimeSegment
    # Event sets to display.
    evtsets: List[tp.EventSet]
    # List of all indexes to display.
    indexes: List[typing.NormalizedIndexKey]
    # The index currently beeing displayed
    selected_index_value: typing.NormalizedIndexKey
    # Items to display
    display_items: List[DisplayItem] = []

    # Is the query button (left click) pressed?
    query_button_pressed: bool = False
    # Is the navigation button (right click) pressed?
    navigation_button_pressed: bool = False

    # Vertical display position.
    display_y_offset: int = 0
    display_y_scale: float = 1.0

    # If true, timestamps are printed as dates. If false, timestamps as printed
    # a numbers.
    # TODO: Configure.
    is_unix_timestamps: bool = True

    # Number of pixels used to print the feature labels.
    feature_label_width: int

    # Size, in px, of the font.
    font_size: int

    # Color palette to display the features / samplings.
    color_palette: List[Tuple[int, int, int]] = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (196, 156, 148),
        (247, 182, 210),
        (199, 199, 199),
        (219, 219, 141),
        (158, 218, 229),
    ]

    def __init__(
        self, evtsets: List[tp.EventSet], args, create_ui: bool = True
    ):
        # Print the available evensets.
        for evtsets_idx, evtset in enumerate(evtsets):
            print(f"Event-set #{evtsets_idx}:\n{evtset}")

        self.font_size = args.font_size
        self.evtsets = evtsets
        self.compute_statistics()
        if create_ui:
            self.create_ui()
        self.select_display(0, ".*")  # Display the first index value
        if create_ui:
            self.refresh_selection()

    def refresh_selection(self):
        """Updates display after a change to the selected index or features.

        Is called after the user changes the selected index or feature filter.
        """
        self.select_display(
            self.index_selector.current(), self.feature_selector.get()
        )
        self.refresh_display()

    def compute_statistics(self):
        """Computes statistics about the evsets to display.

        Is called once during the initialization phase.
        """

        fake_image = Image.new("RGBA", (10, 10), "black")
        fake_draw = ImageDraw.Draw(fake_image)

        self.feature_label_width = 50  # Minimum label width
        index_set: Set[tp.EventSet] = set()
        for evset in self.evtsets:
            # List the available indexes.
            index_set.update(evset.data.keys())

            # Find the longest feature label to print.
            for feature in evset.schema.features:
                label_width = fake_draw.textlength(feature.name)
                self.feature_label_width = max(
                    self.feature_label_width, int(label_width)
                )
        self.indexes = list(index_set)

        print(f"Found {len(self.indexes)} index values")

    def select_display(self, index_idx: int, feature_regexp: str):
        """Select an index to display."""
        self.selected_index_value = self.indexes[index_idx]
        min_time = 0
        max_time = 0
        is_empty = True
        self.display_items = []

        try:
            feature_filter = re.compile(feature_regexp)
        except re.error as e:
            # The user privided an invalid regex.
            return

        # Build the list of items to display.
        for evtset_idx, evset in enumerate(self.evtsets):
            # Scan evtset schema
            if len(evset.schema.features) == 0:
                if feature_filter.match("sampling"):
                    self.display_items.append(
                        DisplayItem(
                            evtset_idx=evtset_idx,
                            feature_idx=-1,
                            label="sampling",
                            height=20,
                        )
                    )
            else:
                for feature_idx, feature in enumerate(evset.schema.features):
                    height = 20 if feature.dtype in [tp.bool_, tp.str_] else 100
                    if feature_filter.match(feature.name):
                        self.display_items.append(
                            DisplayItem(
                                evtset_idx=evtset_idx,
                                feature_idx=feature_idx,
                                label=feature.name,
                                height=height,
                            )
                        )

            # Find time range of data.
            data = evset.data.get(self.selected_index_value)
            if data is None:
                # This event set does not contains this index.
                continue
            if len(data.timestamps) == 0:
                # There is not events to display
                continue
            if is_empty:
                is_empty = False
                min_time = data.timestamps[0]
                max_time = data.timestamps[-1]
            else:
                min_time = min(min_time, data.timestamps[0])
                max_time = max(max_time, data.timestamps[-1])
        if is_empty:
            min_time = 0
            max_time = 100
        else:
            # Add margin
            if min_time == max_time:
                min_time -= 1
                max_time += 1
            else:
                margin = (max_time - min_time) * 0.02
                min_time -= margin
                max_time += margin

        self.total_time = TimeSegment(min_time, max_time)
        self.display_time = TimeSegment(min_time, max_time)

    def create_ui(self):
        """Creates the UI components.

        Called once during the initialization phrase.
        """
        self.root = tk.Tk()
        self.root.title("Temporian Event-Set Viewer")
        self.root.resizable(True, True)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self.top_frame = tk.Frame(self.root)
        self.top_frame.columnconfigure(0, weight=0)
        self.top_frame.columnconfigure(1, weight=1)

        row = 0
        self.index_selector_label = tk.Label(self.top_frame, text="Index")
        self.index_selector_label.grid(column=0, row=0, sticky=tk.E)

        self.index_selector = ttk.Combobox(
            self.top_frame,
            values=[str(x) for x in self.indexes],
            state="readonly",
        )
        self.index_selector.current(0)
        self.index_selector.grid(column=1, row=0, sticky=tk.EW)

        self.feature_selector_label = tk.Label(self.top_frame, text="Feature")
        self.feature_selector_label.grid(column=0, row=1, sticky=tk.E)

        self.feature_selector = ttk.Entry(self.top_frame)
        self.feature_selector.insert(0, ".*")
        self.feature_selector.grid(column=1, row=1, sticky=tk.EW)

        self.top_frame.grid(column=0, row=row, columnspan=2, sticky=tk.EW)

        row += 1
        self.canvas = tk.Canvas(
            self.root, width=self.window_size.x, height=self.window_size.y
        )
        self.canvas_content = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.grid(column=0, row=row, sticky=tk.NSEW)

        row += 1

        self.index_selector.bind(
            "<<ComboboxSelected>>", self.callback_change_index
        )
        self.feature_selector.bind("<KeyRelease>", self.callback_change_feature)
        self.canvas.bind("<Configure>", self.callback_configure)
        self.canvas.bind("<Motion>", self.callback_mouse_move)
        self.canvas.bind("<MouseWheel>", self.callback_mouse_wheel)
        self.canvas.bind("<ButtonPress>", self.callback_mouse_press)
        self.canvas.bind("<ButtonRelease>", self.callback_mouse_release)

    def callback_configure(self, event):
        """Called when the windows size changes."""
        self.window_size = Vector2i(event.width, event.height)

    def callback_change_feature(self, event):
        """Called when the selected features changes."""
        self.refresh_selection()

    def callback_change_index(self, event):
        """Called when the selected index changes."""
        self.refresh_selection()

    def callback_mouse_move(self, event):
        """Called when the mouse moves."""
        self.mouse_position = Vector2i(event.x, event.y)

        if self.navigation_button_pressed:
            # Move the display.
            delta_x = self.mouse_position.x - self.last_mouse_position.x
            delta_y = self.mouse_position.y - self.last_mouse_position.y

            # Note: This equation does not take into account paddings
            delta_time = -(
                delta_x
                * (self.display_time.end - self.display_time.begin)
                / (self.window_size.x - self.feature_label_width)
            )
            self.display_time.begin += delta_time
            self.display_time.end += delta_time

            self.display_y_offset -= delta_y
            if self.display_y_offset < 0:
                self.display_y_offset = 0
            self.last_mouse_position.copy(self.mouse_position)

        self.refresh_display()

    def callback_mouse_wheel(self, event):
        """Called when the mouse wheel is operated."""
        if event.num > 0:
            self.zoom(True)
        elif event.num < 0:
            self.zoom(False)

    def callback_mouse_press(self, event):
        """Called when a mouse button is pressed."""
        if event.num == 1:
            self.query_button_pressed = True
        elif event.num == 3:
            self.navigation_button_pressed = True

        self.last_mouse_position = Vector2i(event.x, event.y)
        self.refresh_display()

    def callback_mouse_release(self, event):
        """Called when a mouse button is released."""
        if event.num == 1:
            self.query_button_pressed = False
            self.refresh_display()
        elif event.num == 3:
            self.navigation_button_pressed = False
            self.refresh_display()
        elif event.num == 5:
            self.zoom(True)
        elif event.num == 4:
            self.zoom(False)

    def zoom(self, dir: bool):
        """User zooms."""
        if self.navigation_button_pressed:
            self.zoom_features(dir)
        else:
            self.zoom_time(dir)

    def zoom_time(self, dir: bool):
        """Zoom time range."""
        center = (self.display_time.begin + self.display_time.end) / 2
        size = self.display_time.end - self.display_time.begin
        rate = 1.2 if dir else 0.8
        self.display_time.begin = center - rate * size / 2
        self.display_time.end = center + rate * size / 2

        self.refresh_display()

    def zoom_features(self, dir: bool):
        """Zoom features range."""
        rate = 0.8 if dir else 1.2
        self.display_y_scale *= rate
        self.display_y_offset = int(self.display_y_offset * rate)
        self.refresh_display()

    def run(self):
        """Start the TK UI loop."""
        self.root.mainloop()

    def refresh_display(self):
        """Refresh the data display."""
        photo = ImageTk.PhotoImage(self.render_frame())
        self.canvas.itemconfigure(self.canvas_content, image=photo)
        self.imgref = photo  # To prevent garbage collection

        self.root.update_idletasks()

    def render_frame(self):
        """Creates a frame showing the eventsets."""
        image = Image.new("RGB", self.window_size.tuple(), "black")
        draw = ImageDraw.Draw(image, "RGBA")

        padding = 2
        timesminimap_height = 20
        timescursor_height = timesminimap_height

        drawing_width = self.window_size.x - padding * 2
        begin_x = padding

        # Feature values / sampling.
        cur_y = (
            padding * 3
            + timescursor_height
            + timesminimap_height
            - self.display_y_offset
        )

        for item_idx, item in enumerate(self.display_items):
            effective_item_height = int(self.display_y_scale * item.height)

            self.draw_item_content(
                draw,
                BBox2i(
                    begin_x + padding + self.feature_label_width,
                    cur_y,
                    -1,
                    effective_item_height,
                ).set_end_x(drawing_width),
                item,
                item_idx,
            )

            label_bbox = BBox2i(
                begin_x,
                cur_y,
                self.feature_label_width,
                effective_item_height,
            )
            draw.rectangle(label_bbox.rect(), fill="black")
            self.draw_item_label(draw, label_bbox, item)

            cur_y += effective_item_height

        # Time
        cur_y = padding
        draw.rectangle(
            (
                0,
                0,
                self.window_size.x,
                padding * 2 + timescursor_height + timesminimap_height,
            ),
            fill="black",
        )

        self.draw_timesminimap(
            draw,
            BBox2i(
                begin_x + padding + self.feature_label_width,
                cur_y,
                -1,
                timesminimap_height,
            ).set_end_x(drawing_width),
        )
        cur_y += timesminimap_height + padding

        self.draw_timescursor(
            draw,
            BBox2i(
                begin_x + padding + self.feature_label_width,
                cur_y,
                -1,
                timesminimap_height,
            ).set_end_x(drawing_width),
        )
        cur_y += timescursor_height + padding

        return image

    def draw_timesminimap(self, draw: ImageDraw.ImageDraw, loc: BBox2i):
        """Draw the time minimap."""
        mid_y = loc.y + loc.h // 2
        draw.line(((loc.x, mid_y), (loc.x + loc.w, mid_y)), fill="white")

        bar_half_height = 5
        x1 = loc.x + loc.w * clamp01(
            self.total_time.av(self.display_time.begin),
        )
        x2 = loc.x + loc.w * clamp01(
            self.total_time.av(self.display_time.end),
        )
        draw.rectangle(
            (x1, mid_y - bar_half_height, x2, mid_y + bar_half_height),
            outline="white",
            fill="black",
        )

    def draw_timescursor(self, draw: ImageDraw.ImageDraw, loc: BBox2i):
        """Draw the time cursor window."""
        draw.line(((loc.x, loc.y), (loc.x + loc.w, loc.y)), fill="white")

        tick_interval_px = 200
        num_ticks = int(self.window_size.x / tick_interval_px)
        tick_length_px = 3

        for tick_idx in range(num_ticks):
            av = (tick_idx + 0.5) / num_ticks
            timestamp = (
                self.display_time.begin
                + (self.display_time.end - self.display_time.begin) * av
            )
            x = loc.x + loc.w * av
            draw.line((x, loc.y, x, loc.y + tick_length_px), fill="white")
            draw.text(
                (x, loc.y + 12),
                self.timestamps_to_str(timestamp),
                anchor="mm",
                fill="white",
                font_size=self.font_size,
            )

        if (
            self.mouse_position.x >= loc.x
            and self.mouse_position.x < loc.x + loc.w
        ):
            mouse_timestamp = self.x_screen_to_timestamp(
                self.mouse_position.x, loc
            )

            if self.query_button_pressed:
                last_mouse_timestamp = self.x_screen_to_timestamp(
                    self.last_mouse_position.x, loc
                )
                diff_seconds = mouse_timestamp - last_mouse_timestamp
                diff_str = str(datetime.timedelta(seconds=abs(diff_seconds)))

                low_x = min(self.mouse_position.x, self.last_mouse_position.x)
                high_x = max(self.mouse_position.x, self.last_mouse_position.x)
                draw.rectangle(
                    (
                        low_x,
                        loc.y,
                        high_x,
                        loc.y + self.window_size.y,
                    ),
                    fill=(255, 0, 0, 127),
                )

                draw.text(
                    (
                        (self.last_mouse_position.x + self.mouse_position.x)
                        / 2,
                        self.last_mouse_position.y + 3,
                    ),
                    f"length: {diff_str}",
                    anchor="mt",
                    fill="red",
                    font_size=self.font_size,
                )

            draw.line(
                (
                    self.mouse_position.x,
                    loc.y,
                    self.mouse_position.x,
                    self.window_size.y,
                ),
                fill="white",
            )
            draw.text(
                (self.mouse_position.x + 3, self.mouse_position.y - 3),
                self.timestamps_to_str(mouse_timestamp),
                anchor="lb",
                fill="white",
                font_size=self.font_size,
            )

    def draw_item_label(
        self, draw: ImageDraw.ImageDraw, loc: BBox2i, item: DisplayItem
    ):
        """Draws the label of a feature / sampling."""
        draw.text(
            (loc.x, loc.y + loc.h // 2),
            item.label,
            anchor="lm",
            font_size=self.font_size,
        )

    def draw_item_content(
        self,
        draw: ImageDraw.ImageDraw,
        loc: BBox2i,
        item: DisplayItem,
        item_idx: int,
    ):
        """Draws the content of a feature / sampling."""
        color = self.color_palette[item_idx % len(self.color_palette)]

        if loc.y > self.window_size.y or loc.y + loc.h < 0:
            # Outside of visible window.
            return

        evtset = self.evtsets[item.evtset_idx]
        data = evtset.data.get(self.selected_index_value)

        if data is None:
            # The index does not exist in this eventset.
            draw.text(
                loc.center().tuple(),
                "*no inde*",
                anchor="mm",
                fill=color,
                font_size=self.font_size,
            )
            return

        if len(data.timestamps) == 0:
            # The eventset has no events for this index.
            draw.text(
                loc.center().tuple(),
                "*no events*",
                anchor="mm",
                fill=color,
                font_size=self.font_size,
            )
            return

        # Finds the range of events to display.
        begin_event_idx = max(
            0,
            np.searchsorted(data.timestamps, self.display_time.begin) - 1,
        )
        end_event_idx = min(
            np.searchsorted(data.timestamps, self.display_time.end) + 1,
            len(data.timestamps),
        )
        timestamps = data.timestamps[begin_event_idx:end_event_idx]

        # Compute the on-screen position.
        screen_x_timestamps = loc.x + loc.w * (
            timestamps - self.display_time.begin
        ) / (self.display_time.end - self.display_time.begin)

        mid_h = loc.y + loc.h // 2
        if item.feature_idx == -1:
            # Display the sampling.
            draw.point(
                [(x, mid_h) for x in screen_x_timestamps],
                fill=color,
            )
            return

        values = data.features[item.feature_idx][begin_event_idx:end_event_idx]

        if np.issubdtype(values.dtype, np.number):
            # Numerical feature.

            # Range of displayed values.
            min_value = np.min(values)
            max_value = np.max(values)
            if min_value != max_value:
                margin = (max_value - min_value) * 0.05
                min_value -= margin
                max_value += margin
                screen_y_values = loc.y + (values - max_value) * loc.h / (
                    min_value - max_value
                )
            else:
                screen_y_values = np.full(len(values), loc.y + loc.h / 2)
                min_value -= 1
                max_value += 1

            # Value ticks.
            ticks = self.gen_ticks(min_value, max_value, loc.h)
            tick_width_px = 2
            for tick in ticks:
                y = loc.y + loc.h * (
                    1.0 - (tick - min_value) / (max_value - min_value)
                )
                draw.line((loc.x, y, loc.x - tick_width_px, y))
                draw.text(
                    (loc.x + 3, y),
                    f"{tick:.4g}",
                    anchor="lm",
                    fill="white",
                    font_size=self.font_size,
                )

            if len(screen_x_timestamps) > 100:
                # Display values as a curve.
                draw.line(
                    list(zip(screen_x_timestamps, screen_y_values)), fill=color
                )
            else:
                # Display values as a cloud of points.
                draw.point(
                    list(zip(screen_x_timestamps, screen_y_values)), fill=color
                )

            if loc.contains(self.mouse_position):
                # The mouse is over this feature.

                # Find the value under the mouse.
                mouse_timestamp = self.x_screen_to_timestamp(
                    self.mouse_position.x, loc
                )
                value_idx = np.searchsorted(timestamps, mouse_timestamp)
                if value_idx >= len(timestamps):
                    value_idx -= 1
                elif value_idx > 0 and (
                    timestamps[value_idx] - mouse_timestamp
                ) > (mouse_timestamp - timestamps[value_idx - 1]):
                    value_idx -= 1

                if value_idx >= 0 and value_idx < len(timestamps):
                    # Print details about value under the mouse.
                    x = screen_x_timestamps[value_idx]
                    y = screen_y_values[value_idx]
                    draw.line(
                        (x - 5, y, x + 5, y),
                        fill="red",
                    )
                    draw.line(
                        (x, y - 5, x, y + 5),
                        fill="red",
                    )

                    draw.text(
                        (x + 3, y + 2),
                        self.timestamps_to_str(timestamps[value_idx]),
                        anchor="lt",
                        fill="red",
                        font_size=self.font_size,
                    )

                    draw.text(
                        (x + 3, y + 12 + 2),
                        f"{values[value_idx]:.5g}",
                        anchor="lt",
                        fill="red",
                        font_size=self.font_size,
                    )

                # Horizontal line.
                draw.line(
                    (
                        loc.x,
                        self.mouse_position.y,
                        loc.x + loc.w,
                        self.mouse_position.y,
                    ),
                    fill="white",
                )

        elif np.issubdtype(values.dtype, np.bool_):
            # Boolean feature
            screen_y_values = loc.y + loc.h * (0.9 - values * 0.8)
            draw.line(
                list(zip(screen_x_timestamps, screen_y_values)), fill=color
            )
        else:
            # Other features. Currently only for string features.
            for x, v in zip(screen_x_timestamps, values):
                draw.text(
                    (x, mid_h),
                    v.decode("utf8"),
                    fill=color,
                    align="center",
                    anchor="mm",
                    font_size=self.font_size,
                )

        draw.rectangle(loc.rect(), outline="white")

    def timestamps_to_str(self, timestamp: float) -> str:
        """String representation of a timestamp."""
        if self.is_unix_timestamps:
            return datetime.datetime.fromtimestamp(
                timestamp, tz=datetime.timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S")
        else:
            return f"{timestamp:.5g}"

    def gen_ticks(
        self, min_value: float, max_value: float, height: int
    ) -> List[float]:
        """Generates the ticks for a numerical feature."""
        num_ticks = height // 25
        ticks = []
        for idx in range(num_ticks):
            av = (idx + 0.5) / num_ticks
            ticks.append(min_value + av * (max_value - min_value))
        return ticks

    def x_screen_to_timestamp(self, x: int, loc: BBox2i) -> float:
        """Timestamp of a x coordinate."""
        return (x - loc.x) * (
            self.display_time.end - self.display_time.begin
        ) / loc.w + self.display_time.begin


def main(args, evtsets: List[tp.EventSet]):
    """Starts the viewer."""
    viewer = Viewer(evtsets, args)
    viewer.run()


def clamp(value, min_value, max_value):
    return min(max_value, max(min_value, value))


def clamp01(value):
    return clamp(value, 0, 1)
