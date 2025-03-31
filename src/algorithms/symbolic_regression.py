import numpy as np
from scipy.stats import norm, rankdata


def xicor(x, y, ties="auto"):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = len(y)

    if len(x) != n:
        raise IndexError(f"x, y length mismatch: {len(x)}, {len(y)}")

    if ties == "auto":
        ties = len(np.unique(y)) < n
    elif not isinstance(ties, bool):
        raise ValueError(
            f'expected ties either "auto" or boolean, '
            f"got {ties} ({type(ties)}) instead"
        )

    y = y[np.argsort(x)]
    r = rankdata(y, method="ordinal")
    nominator = np.sum(np.abs(np.diff(r)))

    if ties:
        max_ranks = rankdata(y, method="max")
        denominator = 2 * np.sum(max_ranks * (n - max_ranks))
        nominator *= n
    else:
        denominator = np.power(n, 2) - 1
        nominator *= 3

    statistic = 1 - nominator / denominator  # upper bound is (n - 2) / (n + 1)
    p_value = norm.sf(statistic, scale=2 / 5 / np.sqrt(n))

    return statistic, p_value
