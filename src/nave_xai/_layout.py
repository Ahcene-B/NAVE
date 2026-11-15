# SPDX-License-Identifier: MIT
"""Common geometry of the summary figures: one fixed width, the same margins, so rows line up."""
FIG_WIDTH = 11.0          # inches, every summary
LEFT, RIGHT = 0.11, 0.89  # axes span (figure fractions): row labels on the left, colour bars on the right
PER_ROW = 10              # concepts per row at most


def columns(K):
    """Concept columns per row: all concepts when there are at most PER_ROW, so they fill the width."""
    return max(1, min(K, PER_ROW))
