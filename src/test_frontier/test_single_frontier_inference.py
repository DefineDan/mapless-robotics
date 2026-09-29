
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

from external.OpenFrontier.frontier.detector import FrontierDetector
from external.OpenFrontier.frontier.model.predict import load_model
from external.OpenFrontier.utils.frontier_utils import read_config_yaml
from external.OpenFrontier.mono_depth.UniK3D import metric_depth_from_rgb

def main():


    img = cv2.imread('src/test_frontier/test.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Random camera parameters
    K = np.array([
        [500.0,   0.0, 320.0],
        [  0.0, 500.0, 240.0],
        [  0.0,   0.0,   1.0],
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = read_config_yaml('external/OpenFrontier/config/navigation.yaml')

    depth = metric_depth_from_rgb(img, K.astype(np.float32))
    # depth = metric_depth_from_rgb(img, None)



    # OpenFrontier
    unet = load_model(
        path=Path("external/OpenFrontier/model_weights/rgbd_11cls.pth"),
        num_classes=config["num_classes"],
        use_depth=True,
    )

    ft_detector = FrontierDetector(
        model=unet,
        camera_intrinsic=K,
        use_depth=True,
        img_size_model=config["input_img_size"],
        device=device,
        )

    ft_detector.detect(rgb=cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                       depth=depth)

    ft_list = ft_detector.anchor_fts(depth=depth, extrinsic=np.eye(4))


    depth_vis = np.clip((255.0 * depth / np.max(depth)).astype(np.uint8), 0, 255)
    depth_vis = cv2.merge((depth_vis, depth_vis, depth_vis))

    h, w = depth_vis.shape[:2]
    size_vec = 20

    for ft in ft_list:

        x, y = ft.pixel_pos
        dx, dy, dz = ft.view_direction

        x = x * w
        y = y * h

        pos_ini = (int(x), int(y)) 
        pos_fin = (int(x + dx * size_vec), int(y + dy * size_vec))
        cv2.line(depth_vis, pos_ini, pos_fin, (0, 255, 0), 2)
        cv2.circle(depth_vis, pos_fin, 3, (0, 0, 255), -1)

    cv2.imwrite('test_depth.png', depth_vis)



if __name__=="__main__":
    main()