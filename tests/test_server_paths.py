"""The one endpoint that turns a browser request into a shell action.

Called without TestClient on purpose: the guard is what matters here, and
testing it directly keeps the suite free of an extra HTTP dependency.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from unmark import server


def test_reveal_refuses_paths_outside_the_output_folder():
    """The browser can ask for any path; only outputs may be revealed."""
    with pytest.raises(HTTPException) as raised:
        server.open_folder(path="C:/Windows/System32/drivers/etc/hosts")
    assert raised.value.status_code == 400
    assert "output folder" in raised.value.detail


def test_reveal_refuses_a_traversal_out_of_the_output_folder():
    escape = str(server.OUTPUT / ".." / "unmark" / "server.py")
    with pytest.raises(HTTPException) as raised:
        server.open_folder(path=escape)
    assert raised.value.status_code == 400


def test_a_missing_output_is_reported_not_revealed():
    with pytest.raises(HTTPException) as raised:
        server.open_folder(path=str(server.OUTPUT / "nothing_here.png"))
    assert raised.value.status_code == 404
