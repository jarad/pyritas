# PyRitas

How to run

Install [dependencies](https://python-poetry.org/docs/basic-usage/#installing-dependencies)

```bash
poetry install
```

Start shell

```bash
poetry shell
```

run tests

```bash
poetry run pytest
```

## Examples

Sample data

```bash
python src/ritas/main.py data/sample.csv "epsg:26915"
```

UIowa data

```bash
python src/ritas/main.py data/Freddies_2023.csv "+proj=longlat +datum=WGS84 +no_defs"
```
