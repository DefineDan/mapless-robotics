
import os
import sys
import torch
import numpy as np
from pathlib import Path
import cv2

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

sys.path.append(os.path.join(PROJECT_ROOT, "external"))
sys.path.append(os.path.join(PROJECT_ROOT, "external", "OpenFrontier"))

from external.OpenFrontier.utils.frontier_utils import read_config_yaml
from src.mapless_robotics.frontier import DepthEstimator, FrontierExtractor


def main():

    img = cv2.imread("src/test_frontier/test.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0],
    ])

    config = read_config_yaml(
        "external/OpenFrontier/config/navigation.yaml"
    )

    depth_estimator = DepthEstimator(K)
    frontier_extractor = FrontierExtractor(config, K)

    depth = depth_estimator(img)

    ft_list = frontier_extractor(
        rgb_input=img,
        depth=depth,
        extrinsic_camera_pos=np.eye(4),
    )

    depth_vis = np.clip(
        255.0 * depth / np.max(depth),
        0,
        255,
    ).astype(np.uint8)

    depth_vis = cv2.merge((depth_vis, depth_vis, depth_vis))

    h, w = depth_vis.shape[:2]
    size_vec = 20

    for ft in ft_list:
        x, y = ft.pixel_pos
        dx, dy, dz = ft.view_direction

        x *= w
        y *= h

        pos_ini = (int(x), int(y))
        pos_fin = (int(x + dx * size_vec), int(y + dy * size_vec))

        cv2.line(depth_vis, pos_ini, pos_fin, (0, 255, 0), 2)
        cv2.circle(depth_vis, pos_fin, 3, (0, 0, 255), -1)

    cv2.imwrite("test_depth.png", depth_vis)


if __name__ == "__main__":
    main()