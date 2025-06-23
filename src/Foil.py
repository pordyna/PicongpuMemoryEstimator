from typing import Mapping
import numpy as np


class Foil:
    def __init__(
        self,
        y_0: float,
        d: float,
        particles_per_cell: Mapping[str, int],
    ):
        self.y_0 = y_0
        self.d = d
        self.particles_per_cell = particles_per_cell

    def __call__(self, species_name, x, y, z=[0,]) -> int:
        y = np.array(y)
        filled_cells_line = np.sum((y>self.y_0) & (y<self.y_0+self.d))
        return int(filled_cells_line * len(z) * len(x) * self.particles_per_cell[species_name])
