
# Class definition to extract frontiers and get depth image if necessary

from pathlib import Path
import numpy as np
import torch

from OpenFrontier.frontier.detector import FrontierDetector
from OpenFrontier.frontier.model.predict import load_model
from unik3d.models import UniK3D
from unik3d.utils.camera import OPENCV 

class FrontierProcess:

    def __init__(self,
                 config:dict,
                 intrinsic_params:np.ndarray,
                 enable_depth_calculation:bool,
                 model_path:str='external/OpenFrontier/model_weights/rgbd_11cls.pth',
                 ):

        self.enable_depth_calculation = enable_depth_calculation 
        self.intrinsic_params = intrinsic_params

        # Read openfrontier 
        unet = load_model(
            path=Path(model_path),
            num_classes=config["num_classes"],
            use_depth=True,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize frontier detector
        self.ft_detector = FrontierDetector(
            model=unet,
            camera_intrinsic=intrinsic_params,
            use_depth=True,
            img_size_model=config["input_img_size"],
            device=self.device,
            )

        # If necessary load model depth
        if self.enable_depth_calculation:
            # TODO, see other depth sizes
            self.model_depth = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl")  # vitl for ViT-L backbone
            self.model_depth = self.model_depth.to(self.device)

            cam_params = [
                intrinsic_params[0, 0],  # fx
                intrinsic_params[1, 1],  # fy
                intrinsic_params[0, 2],  # cx
                intrinsic_params[1, 2],  # cy
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
            params = torch.tensor(cam_params).to(self.device)
            self.camera = OPENCV(params=params)


    def __call__(self, rgb_input:np.ndarray, extrinsic_camera_pos:np.ndarray, depth:np.ndarray=None):

        assert isinstance(depth, np.ndarray) or self.enable_depth_calculation 

        if self.enable_depth_calculation:

            rgb = (
                torch.from_numpy(np.array(rgb_input).astype(np.float32))
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
            )

            depth_predictions = self.model_depth.infer(rgb, self.camera)
            depth = depth_predictions["depth"].squeeze().cpu().numpy()

        # Process frontier detection
        self.ft_detector.detect(rgb=rgb_input, depth=depth)
        ft_list = self.ft_detector.anchor_fts(depth=depth, extrinsic=extrinsic_camera_pos)

        return ft_list