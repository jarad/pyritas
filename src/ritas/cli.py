"""Command line interface for ritas."""

import click


@click.command(help=__doc__)
def main() -> None:
    """Run the command line interface for ritas."""
    click.echo("Hello, world!")
