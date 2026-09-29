

from pathlib import Path
import numpy as np
import torch

from unik3d.models import UniK3D
from unik3d.utils.camera import OPENCV
from OpenFrontier.frontier.detector import FrontierDetector
from OpenFrontier.frontier.model.predict import load_model


class DepthEstimator:

    def __init__(
        self,
        intrinsic_params: np.ndarray,
        model_name: str = "lpiccinelli/unik3d-vitl",
    ):
        self.intrinsic_params = intrinsic_params.astype(np.float32)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load depth model
        self.model = UniK3D.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Camera parameters
        cam_params = [
            self.intrinsic_params[0, 0],  # fx
            self.intrinsic_params[1, 1],  # fy
            self.intrinsic_params[0, 2],  # cx
            self.intrinsic_params[1, 2],  # cy
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        params = torch.tensor(
            cam_params,
            dtype=torch.float32,
            device=self.device,
        )

        self.camera = OPENCV(params=params)

    @torch.no_grad()
    def __call__(self, rgb_input: np.ndarray) -> np.ndarray:

        rgb = (
            torch.from_numpy(rgb_input.astype(np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        depth_predictions = self.model.infer(rgb, self.camera)

        depth = depth_predictions["depth"].squeeze().cpu().numpy()

        return depth



class FrontierExtractor:

    def __init__(
        self,
        config: dict,
        intrinsic_params: np.ndarray,
        model_path: str = "external/OpenFrontier/model_weights/rgbd_11cls.pth",
    ):
        self.intrinsic_params = intrinsic_params.astype(np.float32)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load OpenFrontier model
        unet = load_model(
            path=Path(model_path),
            num_classes=config["num_classes"],
            use_depth=True,
        )

        # Initialize frontier detector
        self.detector = FrontierDetector(
            model=unet,
            camera_intrinsic=self.intrinsic_params,
            use_depth=True,
            img_size_model=config["input_img_size"],
            device=self.device,
        )

    def __call__(
        self,
        rgb_input: np.ndarray,
        depth: np.ndarray,
        extrinsic_camera_pos: np.ndarray,
    ):
        # Detect frontiers
        self.detector.detect(
            rgb=rgb_input,
            depth=depth,
        )

        # Extract frontier anchors
        ft_list = self.detector.anchor_fts(
            depth=depth,
            extrinsic=extrinsic_camera_pos,
        )

        return ft_list