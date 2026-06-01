from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class Vector2D:
    x: float
    y: float

    def __add__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vector2D:
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> Vector2D:
        return self * scalar

    def norm(self) -> float:
        return hypot(self.x, self.y)

    def normalized(self) -> Vector2D:
        length = self.norm()
        if length == 0:
            return Vector2D(0.0, 0.0)
        return Vector2D(self.x / length, self.y / length)

    def distance_to(self, other: Vector2D) -> float:
        return (self - other).norm()

    def dot(self, other: Vector2D) -> float:
        return self.x * other.x + self.y * other.y

    def perpendicular(self) -> Vector2D:
        return Vector2D(-self.y, self.x)

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)
