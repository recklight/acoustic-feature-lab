"""A small published dataset for demonstrating regression diagnostics.

The Portland cement hardening data (Woods, Steinour & Starke, 1932; reprinted
by Hald, 1952) relate the heat evolved during hardening (calories per gram)
to the percentages of four clinker compounds in 13 cement samples. Because
the four percentages nearly add up to 100, the predictors are almost exactly
collinear, which makes the data the classic test case for stepwise selection
and variance inflation factors. The examples and doctests use them, and
``tests/test_reference_data.py`` checks the textbook regression results on
them; every other test runs on synthetic data.
"""

from __future__ import annotations

import pandas as pd

#: Percentages of the four compounds and the evolved heat, one tuple per sample.
_CEMENT_ROWS: tuple[tuple[float, float, float, float, float], ...] = (
    (7, 26, 6, 60, 78.5),
    (1, 29, 15, 52, 74.3),
    (11, 56, 8, 20, 104.3),
    (11, 31, 8, 47, 87.6),
    (7, 52, 6, 33, 95.9),
    (11, 55, 9, 22, 109.2),
    (3, 71, 17, 6, 102.7),
    (1, 31, 22, 44, 72.5),
    (2, 54, 18, 22, 93.1),
    (21, 47, 4, 26, 115.9),
    (1, 40, 23, 34, 83.8),
    (11, 66, 9, 12, 113.3),
    (10, 68, 8, 12, 109.4),
)


def cement_hardening_heat() -> pd.DataFrame:
    """The 13 cement samples: compound percentages and heat of hardening.

    Columns ``x1`` (tricalcium aluminate), ``x2`` (tricalcium silicate),
    ``x3`` (tetracalcium alumino ferrite), ``x4`` (dicalcium silicate), all in
    percent by weight, and ``heat`` (calories per gram).

    Returns
    -------
    pandas.DataFrame
        13 rows, 5 columns.

    References
    ----------
    .. [1] H. Woods, H. H. Steinour, and H. R. Starke, "Effect of composition of
       Portland cement on heat evolved during hardening," Industrial & Engineering
       Chemistry, vol. 24, no. 11, pp. 1207-1214, 1932.
    .. [2] A. Hald, Statistical Theory with Engineering Applications. New York,
       NY, USA: Wiley, 1952.

    Examples
    --------
    >>> data = cement_hardening_heat()
    >>> data.shape, float(data["heat"].mean().round(3))
    ((13, 5), 95.423)
    """
    return pd.DataFrame(list(_CEMENT_ROWS), columns=["x1", "x2", "x3", "x4", "heat"]).astype(float)
