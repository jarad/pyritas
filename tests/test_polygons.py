from typing import Any

import numpy as np
from ritas.polygons import make_bounding_box


def test_make_bounding_box() -> Any:
    x = np.array([1, 3, 5])
    y = np.array([1, 2, 1])
    w = np.array([2, 2, 2])
    d = np.array([np.NaN, 3, 3])

    df = make_bounding_box(x, y, w, d)

    assert df.shape[1] == 12, "Output DataFrame should have 12 columns."
    assert df.shape[0] == len(x), "Number of rows should match the input length."

    # Add more specific assertions as required
