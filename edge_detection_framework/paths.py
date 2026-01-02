import numpy as np

_left = lambda a: (0, 223 - np.tan(a) * 112)
_top = lambda a, height: (112 - height / np.tan(a), 223 - height)
_right = lambda a: (223, 223 + np.tan(a) * 111)


def _get_targets(angle, height, skip_lines=0):
    left_end = np.pi / 2 - np.arctan(112 / height)
    top_end = np.pi - left_end

    intercept = (
        lambda a: _left(a)
        if a < left_end
        else _top(a, height)
        if a < top_end
        else _right(a)
    )

    step = angle * (np.pi / 180)  # convert degrees to radians

    return [
        intercept(a)
        for a in np.arange(step + step * skip_lines, np.pi - step * skip_lines, step)
    ]


def _line_to(source, target):
    length = int(np.hypot(target[0] - source[0], target[1] - source[1]))
    return np.linspace(source, target, length, dtype=int)


def _get_distance(source, target):
    return np.sqrt((target[0] - source[0]) ** 2 + (target[1] - source[1]) ** 2)


def _get_line(source, target):
    l = _line_to(source, target)
    _, idx = np.unique(l, axis=0, return_index=True)
    return l[np.sort(idx)].T


def get_paths(source=(112, 223), height=200, num_lines=14, radius_from=10, skip_lines=0):
    """`get_paths`: Returns paths for the given angle

    Args:
        source  (tuple, optional): x, y coordinates for the origin of our lines. Defaults to (112, 223).
        height    (int, optional): The height of the rectangle within which the lines are drawn. Defaults to 200.
        num_lines (int, optional): The angle step between lines. Must divide 180 evenly. Defaults to 12.

    Returns:
        (tuple[list[np.ndarray], list[np.ndarray], list[str]]):
            lines: NumPy arrays with coordinates for each line,
            distances: NumPy arrays with distance from origin at each corresponding coordinate,
            names: Names of each line
    """
    if num_lines < 1:
        raise ValueError("Must be more than 1 line")
    if skip_lines >= num_lines // 2:
        raise ValueError("Cannot skip all lines")

    targets = _get_targets(180 / (num_lines + 1), height, skip_lines)

    lines = [_get_line(source, t)[..., radius_from:] for t in targets]
    distances = [_get_distance(source, l) for l in lines]
    names = [f"({t[0]:3.0f}, {t[1]:3.0f})" for t in targets]
    return lines, distances, names
