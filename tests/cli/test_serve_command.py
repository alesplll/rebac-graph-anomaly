"""The serve command builds an application without starting a server."""

from pathlib import Path

from rga.cli.main import _parser


def test_serve_is_a_known_command() -> None:
    arguments = _parser().parse_args(
        ["serve", "--config", "configs/service/synthetic.yaml", "--port", "9001"]
    )

    assert arguments.command == "serve"
    assert arguments.config == Path("configs/service/synthetic.yaml")
    assert arguments.port == 9001


def test_serve_defaults_to_a_local_port() -> None:
    arguments = _parser().parse_args(["serve", "--config", "configs/service/synthetic.yaml"])

    assert arguments.host == "127.0.0.1"
    assert arguments.port == 8000
