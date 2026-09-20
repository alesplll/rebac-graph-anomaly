"""The model must never learn from what the analyst decided.

A judgement in the triage journal is a human opinion about the very changes the
model ranks. Any path by which it reaches training turns every measured number into
self-confirmation, so the absence of that path is checked mechanically rather than
promised in prose.
"""

import ast
from pathlib import Path

import pytest

MODEL_PACKAGES = ("src/rga/nn", "src/rga/features", "src/rga/baselines")


def _imports(path: Path) -> set[str]:
    """Every module name this file imports, however it spells the import."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


@pytest.mark.parametrize("package", MODEL_PACKAGES)
def test_the_model_never_imports_the_service(package: str) -> None:
    for path in sorted(Path(package).rglob("*.py")):
        offending = {name for name in _imports(path) if name.startswith("rga.service")}
        assert not offending, f"{path} imports {offending}"


def test_every_guarded_package_was_actually_looked_at() -> None:
    """A glob that matches nothing would make the guard above vacuously true."""
    for package in MODEL_PACKAGES:
        assert list(Path(package).rglob("*.py")), package


def test_the_guard_would_notice(tmp_path: Path) -> None:
    """A test that cannot fail is not a test."""
    path = tmp_path / "leaky.py"
    path.write_text("from rga.service.triage import TriageStore\n", encoding="utf-8")

    assert {name for name in _imports(path) if name.startswith("rga.service")}
