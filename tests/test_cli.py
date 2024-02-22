"""Test the CLI module."""

from click.testing import CliRunner
from ritas.cli import main


def test_cli() -> None:
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input",
            "tests/data/square_100m.csv",
            "--output",
            "tests/data/square_100m.shp",
        ],
    )
    assert result.exit_code == 0
