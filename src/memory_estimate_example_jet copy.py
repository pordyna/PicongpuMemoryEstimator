#!/usr/bin/env python


import math
import numpy as np

import pprint

from MemoryUsageEstimator import MemoryUsageEstimator
from Foil import Foil

"""
@file: memory_estimate_example_jet.py

Estimate the memory usage of a simulation with a jet profile.
The jet axis is the simulation z axis.
"""

cell_size = [
        9.128709291752768e-09,
        9.128709291752768e-09,
        9.128709291752768e-09
    ]
grid_cells = [3104,2208]
gpus_dist = [2, 2]
super_cell_size = [16, 16]
grid_dist = None #[[384, 128, 128, 384], [384, 128, 128, 384], [32] * 8]
jet_radius = 5.0e-6
center_position = [4.7e-6*3, 10.0e-6] # x, y
preplasma_cutoff =50e-9 * 7 # preplasma thickness (not scale length)
jet_radius += preplasma_cutoff
particles_per_cell = {"e": 64, "hydrogen": 64}
ndim=2

if grid_dist is None:
    gpu_cell_extent = np.array(grid_cells) / np.array(gpus_dist)
    grid_extent = [np.ones(gpus_dist[ii]) * gpu_cell_extent[ii] for ii in range(ndim)]
else:
    grid_extent = grid_dist

profile = Foil(
    center_position[0] - jet_radius, 2 * jet_radius, particles_per_cell
)

print(grid_extent)
hists = 0

estimator = MemoryUsageEstimator(
    ndim=ndim,
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
