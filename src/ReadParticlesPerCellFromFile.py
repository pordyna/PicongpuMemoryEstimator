import numpy as np
import scipp as sc


class ReadParticlesPerCellFromFile:
    def __init__(self, data_laoder, iteration, species_names):
        self.dl = data_laoder
        self.mppc_arrays = {}
        for species_name in species_names:
            self.mppc_arrays[species_name] = self.dl.get_field(
                iteration=iteration,
                field=f"{species_name}_all_macroParticleCounter",
                relay=True,
            )

    def __call__(self, species_name, x, y, z) -> int:
        x_extent = (sc.scalar(np.min(x), unit="m"), sc.scalar(np.max(x), unit="m"))
        y_extent = (sc.scalar(np.min(y), unit="m"), sc.scalar(np.max(y), unit="m"))
        z_extent = (sc.scalar(np.min(z), unit="m"), sc.scalar(np.max(z), unit="m"))
        arr = self.mppc_arrays[species_name]
        arr = arr["x", x_extent[0] : x_extent[1]]
        arr = arr["y", y_extent[0] : y_extent[1]]
        arr = arr["z", z_extent[0] : z_extent[1]]
        return sc.sum(arr.load_data()).to(unit="1").value
