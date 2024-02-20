"""Test the CLI module."""

from click.testing import CliRunner
from ritas.cli import main


def test_cli() -> None:
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(main)
    assert result.exit_code == 0
    assert result.output == "Hello, world!\n"
