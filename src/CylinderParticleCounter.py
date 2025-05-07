import numba
from typing import Mapping


@numba.njit(parallel=False)
def _sum_filled_cells_parallel(x, y, max_radius2, center):
    total = 0
    for xx in numba.prange(len(x)):
        for yy in range(len(y)):
            x_rel = x[xx] - center[0]
            y_rel = y[yy] - center[1]
            if (x_rel * x_rel + y_rel * y_rel) <= max_radius2:
                total += 1
    return total


class CylinderParticleCounter:
    def __init__(
        self,
        r: float,
        center: tuple[float, float],
        particles_per_cell: Mapping[str, int],
    ):
        self.max_radius2 = r * r
        self.center = center
        self.particles_per_cell = particles_per_cell

    def __call__(self, species_name, x, y, z) -> int:
        filled_cells_plane = _sum_filled_cells_parallel(
            x, y, self.max_radius2, self.center
        )
        return filled_cells_plane * len(z) * self.particles_per_cell[species_name]
