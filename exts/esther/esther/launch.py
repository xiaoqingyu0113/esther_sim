import argparse
from omni.isaac.lab.app import AppLauncher


def setup():
    print("Setting up Isaac Sim Simulator.")
    # add argparse arguments
    parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    # parse the arguments
    args_cli = parser.parse_args()

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    if not args_cli.headless:
        from omni.isaac.debug_draw import _debug_draw
        debug_drawer = _debug_draw.acquire_debug_draw_interface()
    else:
        debug_drawer = None
    return simulation_app, args_cli, debug_drawer