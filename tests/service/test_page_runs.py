"""The page script, actually executed.

Reading the script as text proves it mentions the right things; it does not prove it
runs. A cached older script against newer markup once died on its first missing
element and left the console inert with nothing on screen to say so. This test runs
the real script against a stub whose `getElementById` refuses unknown identifiers,
so that failure cannot return unnoticed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

HARNESS = Path("tests/service/dom/harness.js")
WEB = Path("web")

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="needs node to execute the page script"
)


def _run() -> dict:
    finished = subprocess.run(
        ["node", str(HARNESS), str(WEB)], capture_output=True, text=True, timeout=60
    )
    assert finished.returncode == 0, finished.stderr
    return json.loads(finished.stdout.strip().splitlines()[-1])


def test_the_script_runs_to_the_end() -> None:
    result = _run()
    assert result["ok"], result.get("error")


def test_loading_the_page_asks_the_service_for_the_queue() -> None:
    """The symptom of the failure was silence: the page requested nothing at all."""
    result = _run()

    assert result["ok"], result.get("error")
    asked = result["asked"]
    assert any(path.startswith("/api/status") for path in asked), asked
    assert any(path.startswith("/api/incidents") for path in asked), asked


def test_the_harness_notices_a_missing_element(tmp_path: Path) -> None:
    """A guard that cannot fail is not a guard."""
    shutil.copy(WEB / "index.html", tmp_path / "index.html")
    (tmp_path / "app.js").write_text(
        'document.getElementById("no-such-thing").hidden = true;\n', encoding="utf-8"
    )

    finished = subprocess.run(
        ["node", str(HARNESS), str(tmp_path)], capture_output=True, text=True, timeout=60
    )
    result = json.loads(finished.stdout.strip().splitlines()[-1])

    assert result["ok"] is False
    assert "no-such-thing" in result["error"]
