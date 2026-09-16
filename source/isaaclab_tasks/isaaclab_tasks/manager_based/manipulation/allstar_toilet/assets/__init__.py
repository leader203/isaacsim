# USD asset path constants resolved relative to this package's data/ directory.
# Layout: allstar_toilet/data/robots/, allstar_toilet/data/scenes/

import os

ALLSTAR_ASSETS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
"""Absolute path to allstar_toilet/data/, containing robot and scene USD files."""
