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
    u = np.array(axis, dtype=np.float64).flatten()
    u /= np.linalg.norm(u)
    ux, uy, uz = u
    cos = np.cos(angle)
    sin = np.sin(angle)
    icos = 1 - cos

    return np.array([
        [cos+ux**2*icos, ux*uy*icos-uz*sin, ux*uz*icos+uy*sin],
        [uy*ux*icos+uz*sin, cos+uy**2*icos, uy*uz*icos-ux*sin],
        [uz*ux*icos-uy*sin, uz*uy*icos+ux*sin, cos+uz**2*icos],
    ])

def Rspin(angle: float):
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )


def Rtilt(angle: float):
    return np.array(
        [
            [(1+np.cos(angle))/2, np.sin(angle)/np.sqrt(2), -(1-np.cos(angle))/2],
            [-np.sin(angle)/np.sqrt(2), np.cos(angle), -np.sin(angle)/np.sqrt(2)],
            [-(1-np.cos(angle))/2, np.sin(angle)/np.sqrt(2), (1+np.cos(angle))/2],
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