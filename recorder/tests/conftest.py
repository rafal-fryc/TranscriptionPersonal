import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir() -> str:
    with tempfile.TemporaryDirectory() as d:
        yield d
