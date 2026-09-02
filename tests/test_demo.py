from pathlib import Path

import pytest


def test_streamlit_app_starts_without_a_loaded_model():
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    app_path = Path(__file__).parents[1] / "demo" / "streamlit_app.py"

    app = streamlit_testing.AppTest.from_file(str(app_path))
    app.run(timeout=60)

    assert not app.exception
