"""Simple statistics utilities."""


def running_mean(values):
    """Return the cumulative running mean for the given values.

    For each position i, the result is the average of values[0..i].
    """
    result = []
    total = 0
    count = 0
    for v in values:
        total += v
        count += 1
        result.append(total / count)
    return result
