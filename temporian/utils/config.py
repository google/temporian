import os

debug_mode = os.environ.get("TEMPORIAN_DEBUG_MODE", False)

# Limits for repr(evset), print(evset)
max_printed_indexes = os.environ.get("TEMPORIAN_MAX_PRINTED_INDEXES", 5)
max_printed_features = os.environ.get("TEMPORIAN_MAX_PRINTED_FEATURES", 10)

# Limits for html display of evsets (notebooks)
# Indexes (number of tables)
max_display_indexes = os.environ.get("TEMPORIAN_MAX_DISPLAY_INDEXES", 10)
# Features (columns) per table
max_display_features = os.environ.get("TEMPORIAN_MAX_DISPLAY_FEATURES", 20)
# Events (rows) per table
max_display_events = os.environ.get("TEMPORIAN_MAX_DISPLAY_EVENTS", 100)
# Chars per table cell
max_display_chars = os.environ.get("TEMPORIAN_MAX_DISPLAY_CHARS", 50)
