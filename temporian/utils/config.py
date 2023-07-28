import os

debug_mode = bool(os.environ.get("TEMPORIAN_DEBUG_MODE", False))

# Limits for repr(evset), print(evset)
max_printed_indexes = int(os.environ.get("TEMPORIAN_MAX_PRINTED_INDEXES", 5))
max_printed_features = int(os.environ.get("TEMPORIAN_MAX_PRINTED_FEATURES", 10))
max_printed_events = int(os.environ.get("TEMPORIAN_MAX_PRINTED_EVENTS", 20))

# Limits for html display of evsets (notebooks)
# Indexes (number of tables)
max_display_indexes = int(os.environ.get("TEMPORIAN_MAX_DISPLAY_INDEXES", 10))
# Features (columns) per table
max_display_features = int(os.environ.get("TEMPORIAN_MAX_DISPLAY_FEATURES", 20))
# Events (rows) per table
max_display_events = int(os.environ.get("TEMPORIAN_MAX_DISPLAY_EVENTS", 100))
# Chars per table cell
max_display_chars = int(os.environ.get("TEMPORIAN_MAX_DISPLAY_CHARS", 50))

# Configs for both repr and html
# Decimal numbers precision
print_precision = int(os.environ.get("TEMPORIAN_PRINT_PRECISION", 4))
