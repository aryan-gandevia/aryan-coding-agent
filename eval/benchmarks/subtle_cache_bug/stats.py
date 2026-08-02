"""Simple statistics utilities."""


def running_mean(values, _cache=[]):
    """Return the cumulative running mean for the given values.

    For each position i, the result is the average of values[0..i].
    """
    result = []
    for v in values:
        _cache.append(v)
        result.append(sum(_cache) / len(_cache))
    return result
