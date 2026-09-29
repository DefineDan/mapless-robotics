

import logging
import argparse
from pathlib import Path
import open3d as o3d


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--target", type=str, required=True, help="Target object to search for"
    )

    p.add_argument("--mesh", type=str, required=True, help="Path to the mesh file")
    p.add_argument(
        "--config",
        type=str,
        default="config/hm3d_navigation.yaml",
        help="OpenFrontier configuration file",
    )
    p.add_argument(
        "--write_path", type=str, help="JSON file to write the ftmanager state"
    )
    p.add_argument(
        "--auto_start",
        action="store_true",
        default=False,
        help="Auto-start navigation loop",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI windows and record navigation_video.mp4",
    )
    p.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum number of navigation steps",
    )
    p.add_argument(
        "--max_time", type=int, default=3600, help="Maximum navigation time in seconds"
    )
    p.add_argument(
        "--vis_graph",
        action="store_true",
        default=False,
        help="Visualize the topological graph",
    )
    p.add_argument(
        "--save-images",
        "--save_images",
        dest="save_images",
        action="store_true",
        default=False,
        help="Save intermediate navigation images",
    )
    p.add_argument(
        "--record-video",
        "--record_video",
        dest="record_video",
        action="store_true",
        default=False,
        help="Record navigation_video.mp4",
    )
    p.add_argument(
        "--image_preview",
        action="store_true",
        default=False,
        help="Write the latest composed video frame to topdown.png",
    )
    p.add_argument(
        "--unet_weight",
        type=Path,
        default=Path("model_weights/rgbd_11cls.pth"),
        help="Path to UNet model weights",
    )
    # p.add_argument(
    #     "--depth_source",
    #     type=str,
    #     default=NavigationApp.DEPTH_GT,
    #     choices=[
    #         NavigationApp.DEPTH_GT,
    #         NavigationApp.DEPTH_M3D,
    #         NavigationApp.DEPTH_UNIK3D,
    #     ],
    #     help="Depth source",
    # )
    p.add_argument(
        "--log_level",
        "-ll",
        type=int,
        default=20,
        help="logging level (0=notset, 10=debug, 20=info...)",
    )
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.headless:
        args.auto_start = True
        args.record_video = True
        args.image_preview = True

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=args.log_level,
    )
    print(f"Open3D version: {o3d.__version__}")

    # app = NavigationApp(args)
    # exit_code = app.run()
    # sys.stdout.flush()
    # sys.stderr.flush()
    # os._exit(exit_code)


if __name__ == "__main__":
    main()
