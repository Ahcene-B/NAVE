# SPDX-License-Identifier: MIT
"""Progress reporting: False (silent), True (text bar on stderr) or a callable(fraction, text)."""
import sys


def make_progress(progress):
    if callable(progress):
        return progress
    if not progress:
        return lambda fraction, text="": None
    state = {"last": ""}

    def bar(fraction, text=""):
        filled = int(round(30 * fraction))
        line = f"\r[{'#' * filled}{'.' * (30 - filled)}] {100 * fraction:5.1f}%  {text}"
        sys.stderr.write(line + " " * max(0, len(state["last"]) - len(line)))
        state["last"] = line
        if fraction >= 1.0:
            sys.stderr.write("\n")
        sys.stderr.flush()
    return bar


def span(report, start, end):
    """Map progress in [0, 1] onto [start, end] of `report`."""
    return lambda fraction, text="": report(start + (end - start) * fraction, text)
