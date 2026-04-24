import pygame
from dataclasses import dataclass, field

@dataclass
class Car:
    lane_index: int
    x: float
    y: float
    width: int
    height: int
    color: tuple[int, int, int] = (255, 255, 255)
    speed: float = 0.0
    image: pygame.Surface = field(default=None)
    passed: bool = False  

    def rect(self) -> pygame.Rect:
        return pygame.Rect(
            int(self.x - self.width / 2),
            int(self.y - self.height / 2),
            self.width,
            self.height,
        )

@dataclass
class HandState:
    found: bool = False
    x_norm: float = 0.5