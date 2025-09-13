# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates procedural terrain generation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBase

import os.path as osp
root = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))

path = osp.join(root, "source")
import sys
sys.path.append(path)


from isaacvln.terrains import terrains_cfg, matterport

ASSETS_DIR = "/workspace/data2/VSCODE/VLN/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/assets"

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.assets import Articulation

def design_scene() -> tuple[dict, torch.Tensor]:
    """Designs the scene."""
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)


    # Handler for terrains importing
    terrain_importer_cfg = terrains_cfg.MatterportImporterCfg(
        env_spacing=3.0,
        prim_path="/World/matterport",
        terrain_type="matterport",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        obj_filepath=osp.join(ASSETS_DIR, "matterport_usd/5q7pvUzZiYa/5q7pvUzZiYa.usd"),
        groundplane=False,
    )

    # Create terrain importer
    terrain_importer = matterport.MatterportImporter(terrain_importer_cfg)
    # return the scene information

    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 0.0, 1)
    robot.spawn.articulation_props.fix_root_link = True
    g1 = Articulation(robot.replace(prim_path="/World/G1"))

    scene_entities = {"terrain": terrain_importer,
                      "robot": g1}

    return scene_entities, terrain_importer.env_origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, AssetBase], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[5.0, 5.0, 5.0], target=[0.0, 0.0, 0.0])
    # design scene
    scene_entities, scene_origins = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
