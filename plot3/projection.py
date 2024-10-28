import numpy as np

from numpy import typing as npt

S = np.array(
    [
        [np.sqrt(3) / 2, 0, -np.sqrt(3) / 2],
        [0.5, 1, 0.5],
    ]
)


def Ry(angle: float):
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )


def Rz(angle: float):
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )


def project(obj: npt.NDArray, tilt: float = 0, yaw: float = 0):
    """Project using isometric projection with arbitrary tilt and yaw

    Args:
        obj (NDArray): object to project
        tilt (float): rotation around the y-axis
        yaw (float): rotation around the z-axis
    """

    return S @ Ry(tilt) @ Rz(yaw) @ obj
