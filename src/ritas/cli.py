"""
Command line interface for RITAS.

This tool is used to convert an input file with yield monitor data to an output
grid file.

"""

from pathlib import Path

import click

from ritas.workflows import simple_workflow


@click.command(help=__doc__)
@click.option(
    "--input", "-i", "infile", type=click.Path(exists=True), required=True
)
@click.option(
    "--output", "-o", "outfile", type=click.Path(exists=False), required=True
)
def main(infile: Path, outfile: Path) -> None:
    """Run the command line interface for ritas."""
    click.echo(f"I am about to process {infile} -> {outfile}")
    simple_workflow(infile, outfile)
