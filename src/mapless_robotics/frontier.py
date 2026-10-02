import sys
from pathlib import Path

import numpy as np
import torch

OPENFRONTIER_ROOT = Path(__file__).resolve().parents[2] / "external" / "OpenFrontier"


def _openfrontier_imports():
    # Upstream uses absolute frontier.* and utils.* imports internally.
    if str(OPENFRONTIER_ROOT) not in sys.path:
        sys.path.insert(0, str(OPENFRONTIER_ROOT))
    from frontier.detector import FrontierDetector
    from frontier.model.predict import load_model

    return FrontierDetector, load_model


class DepthEstimator:

    def __init__(
        self,
        intrinsic_params: np.ndarray,
        model_name: str = "lpiccinelli/unik3d-vitl",
    ):
        from unik3d.models import UniK3D
        from unik3d.utils.camera import OPENCV

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
        model_path: str | Path = OPENFRONTIER_ROOT / "model_weights/rgbd_11cls.pth",
    ):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"OpenFrontier weights not found: {model_path}. Use --frontier-weights."
            )
        FrontierDetector, load_model = _openfrontier_imports()
        self.df_normalizer = config.get("df_normalizer", 3.0)
        self.df_thr = config.get("df_thr", 0.1)
        self.intrinsic_params = intrinsic_params.astype(np.float32)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load OpenFrontier model
        unet = load_model(
            path=Path(model_path),
            num_classes=config["num_classes"],
            use_depth=True,
        )

        unet.eval()

        # Initialize frontier detector
        self.detector = FrontierDetector(
            model=unet,
            camera_intrinsic=self.intrinsic_params,
            use_depth=True,
            img_size_model=config["input_img_size"],
            device=self.device,
        )

    @torch.no_grad()
    def __call__(
        self,
        rgb_input: np.ndarray,
        depth: np.ndarray,
        extrinsic_camera_pos: np.ndarray,
    ):
        """Use an OpenCV world-to-camera extrinsic; return world-frame frontiers."""
        # Detect frontiers
        self.detector.detect(
            rgb=rgb_input,
            depth=depth,
            df_normalizer=self.df_normalizer,
            df_thr=self.df_thr,
        )

        # Extract frontier anchors
        ft_list = self.detector.anchor_fts(
            depth=depth,
            extrinsic=extrinsic_camera_pos,
        )

        return [] if ft_list is None else ft_list

    def visualize(self, frontiers):
        """Draw OpenFrontier's marks on the latest detector crop; return RGB.

        Call immediately after extraction on the inference thread: the upstream
        detector owns mutable image/preprocessing state. Labels are local to
        this capture, not persistent graph frontier IDs.
        """
        for index, frontier in enumerate(frontiers, start=1):
            frontier.label = str(index)
        marked, _, _ = self.detector.get_SoM_img(frontiers, radius=12, alpha=0.55)
        # With no detections upstream returns a PIL image instead of an array.
        return np.array(marked, dtype=np.uint8, copy=True)
