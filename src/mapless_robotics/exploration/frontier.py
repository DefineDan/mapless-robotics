import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

OPENFRONTIER_ROOT = Path(__file__).resolve().parents[3] / "external" / "OpenFrontier"


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
        self.model.resolution_level = 6
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

        # UniK3D mutates the camera during inference (device transfer, crop,
        # and resize). Keep the calibrated camera unchanged across frames.
        depth_predictions = self.model.infer(rgb, self.camera.clone())

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
        mask: np.ndarray | None = None,
    ):
        """Return world-frame frontiers, optionally excluding invalid pixels.

        ``mask`` is a 2D validity mask aligned with ``rgb_input`` (or already
        in processed detector coordinates). A frontier is removed when its
        anchor pixel is zero in the mask.
        """
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

        raw_count = len(ft_list) if ft_list is not None else 0
        mask_coverage = None
        if ft_list is not None and mask is not None:
            mask = np.asarray(mask)
            if mask.ndim != 2:
                raise ValueError("mask must be a 2D array")
            if mask.shape == rgb_input.shape[:2]:
                # Match detector preprocessing, using nearest-neighbor so a
                # binary validity mask stays binary through resize and crop.
                height, width = mask.shape
                resized = Image.fromarray(mask.astype(np.uint8)).resize(
                    (int(width * self.detector.scale_factor),
                     int(height * self.detector.scale_factor)),
                    Image.Resampling.NEAREST,
                )
                model_width, model_height = self.detector.img_size_model
                left = (resized.width - model_width) // 2
                top = (resized.height - model_height) // 2
                mask = np.asarray(
                    resized.crop((left, top, left + model_width, top + model_height))
                )
            if mask.shape != self.detector.ft_region.shape:
                raise ValueError(
                    "mask must match rgb_input or the processed frontier map "
                    f"{self.detector.ft_region.shape}; "
                    f"got {mask.shape}"
                )

            mask_coverage = float(np.mean(mask > 0))
            height, width = mask.shape
            ft_list = [
                frontier
                for frontier in ft_list
                if mask[
                    min(max(int(frontier.pixel_pos[1] * height), 0), height - 1),
                    min(max(int(frontier.pixel_pos[0] * width), 0), width - 1),
                ]
            ]

        self.last_diagnostics = dict(
            raw_anchors=raw_count, retained_anchors=len(ft_list) if ft_list is not None else 0,
            frontier_pixels=int(np.count_nonzero(self.detector.ft_region)),
            valid_depth_fraction=float(np.mean(np.isfinite(depth) & (depth > 0))),
            mask_valid_fraction=mask_coverage,
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
