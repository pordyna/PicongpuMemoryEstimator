from picongpu.extra.utils.memory_calculator import MemoryCalculator
import numpy as np
from typing import Sequence, Mapping, Iterable, Callable


class MemoryUsageEstimator:
    def __init__(
        self,
        ndim: int,
        dx: Sequence[float],
        gpus_dist: Sequence[int],
        precision: int = 32,
        particle_shape_order: int = 3,
        number_of_temporary_field_slots: int = 3,
        custom_attributes_size_dict: Mapping[str, int] = {"collison_estimate": 10},
        species_names: Sequence[str] = ["e", "H"],
        particle_attributes: Mapping[str, list[str]] = {
            "e": ["momentum", "position", "weighting", "collison_estimate"],
            "H": ["momentum", "position", "weighting", "collison_estimate"],
        },
        super_cell_size: Sequence[int] = (8, 8, 4),
        memory_binning_pluggin: int = 0,
        get_number_particle_cells: Callable[
            [str, Sequence[float], Sequence[float], Sequence[float]], int
        ]
        | Callable[[str, Sequence[float], Sequence[float]], int] = lambda x,
        y,
        z=None: len(x) * len(y) * (len(z) if z is not None else 1),
        pml_size: Iterable | None = None,
    ):
        self.ndim = ndim
        self.gpus_dist = gpus_dist[:ndim]
        self.dx = dx[:ndim]
        self.precision = precision
        self.particle_shape_order = particle_shape_order
        self.number_of_temporary_field_slots = number_of_temporary_field_slots
        self.custom_attributes_size_dict = custom_attributes_size_dict
        self.species_names = species_names
        self.particle_attributes = particle_attributes
        self.memory_binning_pluggin = memory_binning_pluggin
        self.super_cell_size = np.array(super_cell_size[:ndim])
        self.get_number_particle_cells = get_number_particle_cells
        if pml_size is None:
            pml_size = ((12, 12), (12, 12), (12, 12))
        self.pml_size = np.array(pml_size)
        self.mc = MemoryCalculator(
            simulation_dimension=self.ndim,
            super_cell_size=self.super_cell_size,
            precision=self.precision,
            particle_shape_order=self.particle_shape_order,
            pml_border_size=self.pml_size,
        )

    def _get_memory_species(self, n_particles: int, species_name: str) -> int:
        """Get the memory required for a species on a GPU

        Args:
            n_particles:  number of particles in the spiecies on the gpu
            species_name: name of the species
        """

        # mem calulcator doues weird stuff and asks for cells and number
        # of particles per cell, instead of number of particles
        # so we just give it one cell with all the prticles
        return self.mc.memory_required_by_particles_of_species(
            particle_filled_cells=np.ones(self.ndim),
            species_attribute_list=self.particle_attributes[species_name],
            particles_per_cell=int(n_particles),
            custom_attributes_size_dict=self.custom_attributes_size_dict,
        )

    def memory_per_gpu(
        self, extent: Iterable[int], particles_on_gpu: Mapping[str, int]
    ) -> Mapping[str, int]:
        extent = np.array(extent)
        field_gpu = self.mc.memory_required_by_cell_fields(
            extent, number_of_temporary_field_slots=self.number_of_temporary_field_slots
        )
        rng_gpu = self.mc.memory_required_by_random_number_generator(extent)
        particle_gpus = 0

        for species_name in self.species_names:
            particle_gpus += self._get_memory_species(
                particles_on_gpu[species_name], species_name
            )

        total = field_gpu + rng_gpu + particle_gpus + self.memory_binning_pluggin
        return {
            "total": total,
            "rng": rng_gpu,
            "fields": field_gpu,
            "particles": particle_gpus,
        }

    def get_estimates(self, cells: Sequence[Sequence[int]]) -> Mapping[str, float]:
        gpu_memory = np.zeros(self.gpus_dist, dtype=int)
        gpu_memory_rng = np.zeros(self.gpus_dist, dtype=int)
        gpu_memory_field = np.zeros(self.gpus_dist, dtype=int)
        gpu_memory_particles = np.zeros(self.gpus_dist, dtype=int)

        n_particles = {species: np.zeros(self.gpus_dist, dtype=int) for species in self.species_names}

        offsets = [
            np.insert(np.cumsum(cells[ii][:-1]), 0, 0) for ii in range(self.ndim)
        ]
        for species_name in self.species_names:
            with np.nditer(
                n_particles[species_name],
                flags=["multi_index",],
                op_flags=[("writeonly",)],
            ) as it:
                for n_particles_it in it:
                    offset = [offsets[ii][it.multi_index[ii]] for ii in range(self.ndim)]
                    extent = [cells[ii][it.multi_index[ii]] for ii in range(self.ndim)]
                    positions = [
                        np.arange(offset[ii], offset[ii] + extent[ii]) * self.dx[ii]
                        for ii in range(self.ndim)
                    ]
                    n_particles_it[...] = self.get_number_particle_cells(
                        species_name, *positions
                    )


        with np.nditer(
            [
                gpu_memory,
                gpu_memory_rng,
                gpu_memory_field,
                gpu_memory_particles,
            ],
            flags=["multi_index"],
            op_flags=[("writeonly",)] * 4,
        ) as it:
            for (
                gpu_memory_it,
                gpu_memory_rng_it,
                gpu_memory_field_it,
                gpu_memory_particles_it,
            ) in it:
                extent = [cells[ii][it.multi_index[ii]] for ii in range(self.ndim)]

                result = self.memory_per_gpu(
                    extent,
                    {k: v[it.multi_index] for k, v in n_particles.items()},
                )
                gpu_memory_it[...] = result["total"]
                gpu_memory_rng_it[...] = result["rng"]
                gpu_memory_field_it[...] = result["fields"]
                gpu_memory_particles_it[...] = result["particles"]

        max_memory_init = np.max(gpu_memory)
        memory_inbalance_init = max_memory_init - np.min(gpu_memory)
        gb_to_byte = 1024**3
        outcome = {
            "max_memory_GiB": float(max_memory_init) / gb_to_byte,
            "memory_inbalance_GiB": float(memory_inbalance_init) / gb_to_byte,
            "max_rng_GiB": float(np.max(gpu_memory_rng)) / gb_to_byte,
            "max_field_GiB": float(np.max(gpu_memory_field)) / gb_to_byte,
            "max_particles_GiB": float(np.max(gpu_memory_particles)) / gb_to_byte,
        }
        return outcome
