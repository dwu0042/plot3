import numpy as np
from matplotlib import pyplot as plt
from .projection import project

from numpy.typing import NDArray
from matplotlib.axes import Axes


def plot(
    xs: NDArray,
    ys: NDArray,
    zs: NDArray,
    ax: Axes | None = None,
    method="line",
    tilt: float = 0,
    yaw: float = 0,
    **kwargs,
):
    xa = np.array(xs)
    ya = np.array(ys)
    za = np.array(zs)

    points = np.vstack(
        [
            xa.reshape((1, -1)),
            ya.reshape((1, -1)),
            za.reshape((1, -1)),
        ]
    )

    flat_points = project(points, tilt=tilt, yaw=yaw)

    if ax is None:
        ax = plt.figure().add_subplot()

    match method:
        case "line":
            ax.plot(*flat_points, **kwargs)
        case "scatter":
            ax.scatter(*flat_points, **kwargs)
        case _:
            raise ValueError(f"Unknown method {method}")

    ax.set_aspect("equal")

    return ax


def scatter(
    xs: NDArray,
    ys: NDArray,
    zs: NDArray,
    ax: Axes | None = None,
    tilt: float = 0,
    yaw: float = 0,
    **kwargs,
):
    return plot(xs, ys, zs, ax=ax, method="scatter", tilt=tilt, yaw=yaw, **kwargs)
