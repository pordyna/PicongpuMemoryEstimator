#!/usr/bin/env python


import math
import numpy as np

import pprint

from MemoryUsageEstimator import MemoryUsageEstimator
from CylinderParticleCounter import CylinderParticleCounter

"""
@file: memory_estimate_example_jet.py

Estimate the memory usage of a simulation with a jet profile.
The jet axis is the simulation z axis.
"""

cell_size = [10e-9, 10e-9, 10e-9]
grid_cells = [1024, 1024, 256]
gpus_dist = [4, 4, 8]
super_cell_size = [8, 8, 4]
grid_dist = [[384, 128, 128, 384], [384, 128, 128, 384], [32] * 8]
jet_radius = 2.4e-6
center_position = [2.0e-6, 2.0e-6] # x, y
preplasma_cutoff = 10e-9 * 10 # preplasma thickness (not scale length)
jet_radius += preplasma_cutoff
particles_per_cell = {"e": 32, "hydrogen": 32}

if grid_dist is None:
    gpu_cell_extent = np.array(grid_cells) / np.array(gpus_dist)
    grid_extent = [np.ones(gpus_dist[ii]) * gpus_dist[ii] for ii in range(3)]
else:
    grid_extent = grid_dist

profile = CylinderParticleCounter(
    jet_radius, center_position, particles_per_cell
)


hists = 0

estimator = MemoryUsageEstimator(
    ndim=3,
    dx=cell_size,
    gpus_dist=gpus_dist,
    get_number_particle_cells=profile,
    super_cell_size=super_cell_size,
    precision=32,
    custom_attributes_size_dict={"collison_estimate": 10},
    species_names=["e", "hydrogen"],
    particle_attributes={
        "e": ["momentum", "position", "weighting", "collison_estimate"],
        "hydrogen": ["momentum", "position", "weighting", "collison_estimate"],
    },
    number_of_temporary_field_slots=3,
    memory_binning_pluggin=math.ceil(hists),
)
estimates = estimator.get_estimates(grid_extent)
pprint.pprint(estimates)
