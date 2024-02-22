# PyRitas

A python implementation of the [R RITAS Algorithm](https://github.com/luisdamiano/ritas-pkg),
which attempts to create smoothed yield maps from raw yield monitor data.

[![PyPI Package](https://img.shields.io/pypi/v/ritas.svg)](https://pypi.python.org/pypi/ritas/)
[![Conda Package](https://anaconda.org/conda-forge/ritas/badges/version.svg)](https://anaconda.org/conda-forge/ritas)
[![PyPI Downloads](https://img.shields.io/pypi/dm/ritas.svg)](https://pypi.python.org/pypi/ritas/)
[![Conda Downloads](https://anaconda.org/conda-forge/ritas/badges/downloads.svg)](https://anaconda.org/conda-forge/ritas)

## Installation

Via PyPI / pip

```bash
python -m pip install ritas
```

Via [conda-forge](https://conda-forge.org)

```bash
conda -c conda-forge install ritas
```

The above will install both the `ritas` python module and a front-end script
called `ritas`, which can be used like so:

## Example Usage

Read an input file called `yielddata.csv` and create an output file called
`output.tiff`.

```bash
ritas -i yielddata.csv -o output.shp
```
