import numpy as np

from typing import Sequence
from numpy import typing as npt

S = np.array(
    [
        [np.sqrt(3) / 2, 0, -np.sqrt(3) / 2],
        [0.5, 1, 0.5],
    ]
)


def Rarb(angle: float, axis: Sequence[float]):
    """Arbitrary rotation around given axis"""

    # normalise axis vector
    u = np.array(axis, dtype=np.float64).flatten()
    u /= np.linalg.norm(u)
    ux, uy, uz = u

    # use Rodrigues' rotation formula
    # R = I + sin(θ)K + (1- cos(θ))K²
    # where K is the cross-product matrix
    K = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0],
    ])

    return np.eye(3) + np.sin(angle) * K + (1- np.cos(angle)) * (K@K)


def Rspin(angle: float):
    """Rotate around the y axis (vertical)"""
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )


def Rtilt(angle: float):
    """Rotate around the anti-(XZ) axis"""
    return np.array(
        [
            [
                (1 + np.cos(angle)) / 2,
                np.sin(angle) / np.sqrt(2),
                -(1 - np.cos(angle)) / 2,
            ],
            [-np.sin(angle) / np.sqrt(2), np.cos(angle), -np.sin(angle) / np.sqrt(2)],
            [
                -(1 - np.cos(angle)) / 2,
                np.sin(angle) / np.sqrt(2),
                (1 + np.cos(angle)) / 2,
            ],
        ]
    )


def project(obj: npt.NDArray, tilt: float = 0, spin: float = 0):
    """Project using isometric projection with arbitrary tilt and yaw

    Args:
        obj (NDArray): object to project
        tilt (float): angle to tilt vertically (radians)
        yaw (float): angle to spin about bottom plane (radians)
    """

    return S @ Rspin(spin) @ Rtilt(tilt) @ obj
