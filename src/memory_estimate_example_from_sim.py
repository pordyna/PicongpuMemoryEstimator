#!/usr/bin/env python

import sys
import json
import math
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm

import openpmd_scipp as pmdsc

from MemoryUsageEstimator import MemoryUsageEstimator
from ReadParticlesPerCellFromFile import ReadParticlesPerCellFromFile

simulation_path = Path(
    "/global/cfs/cdirs/m4251/pordyna/runs/hydrogenTempXRTSscan/3D_PICONGPU/scan06/simulations/sim_2"
)

with open(
    simulation_path / f"../../projects/{simulation_path.name}/pypicongpu.json", "r"
) as infile:
    pypicongpu_dict = json.load(infile)

cell_size = np.array(list(pypicongpu_dict["grid"]["cell_size"].values()))
grid_cells = np.array(list(pypicongpu_dict["grid"]["cell_cnt"].values()))
gpus_dist = np.array(list(pypicongpu_dict["grid"]["gpu_cnt"].values()))
super_cell_size = np.array(list(pypicongpu_dict["grid"]["super_cell_size"].values()))

grid_dist = pypicongpu_dict["grid"].get("grid_dist")
jet_radius = pypicongpu_dict["species_initmanager"]["operations"]["simple_density"][0][
    "profile"
]["data"]["radius_si"]
center_position = pypicongpu_dict["species_initmanager"]["operations"][
    "simple_density"
][0]["profile"]["data"]["center_position_si"]
center_position = [e["component"] for e in center_position]
if not pypicongpu_dict["species_initmanager"]["operations"]["simple_density"][0][
    "profile"
]["data"]["pre_plasma_ramp"]["type"]["none"]:
    jet_radius += pypicongpu_dict["species_initmanager"]["operations"][
        "simple_density"
    ][0]["profile"]["data"]["pre_plasma_ramp"]["data"]["PlasmaCutoff"]

if grid_dist is None:
    gpu_cell_extent = grid_cells / gpus_dist
    grid_extent = [np.ones(gpus_dist[ii]) * gpu_cell_extent[ii] for ii in range(3)]
else:
    grid_dist_x = [e["device_cells"] for e in grid_dist["x"]]
    grid_dist_y = [e["device_cells"] for e in grid_dist["y"]]
    grid_dist_z = [e["device_cells"] for e in grid_dist["z"]]
    grid_extent = [grid_dist_x, grid_dist_y, grid_dist_z]


series_path = simulation_path / "simOutput/openPMD/simData%T.bp5"
output_dir = simulation_path / "post-processing"
output_dir.mkdir(parents=False, exist_ok=True)

dl = pmdsc.DataLoader(series_path)
results = {}
for iteration in tqdm(list(dl.series.iterations)):
    try:
        profile = ReadParticlesPerCellFromFile(
            dl,
            iteration,
            ["e", "hydrogen"],
        )

        hists = (4 * 3 * 1024 * 64 / 8) + (4 * 3 * 1024 * 64 / 8 * 129 * 129 * 3)

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
        results[iteration] = estimates
        print(f"Iteration {iteration}: {estimates}", flush=True)
    except Exception as e:
        print(f"Error in iteration {iteration}: {e}", flush=True, file=sys.stderr)
        continue
with open(output_dir / "memory_estimate.json", "w") as outfile:
    json.dump(results, outfile, indent=4)
