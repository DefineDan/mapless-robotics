
import numpy as np

from OpenFrontier.nav.agent import NavigationAgent
from OpenFrontier.planner.occ_rrt_point3d import OccupancyGrid3DPathPlanner


class AgentV1(NavigationAgent):

    def __init__(self,
                args,
                target: str,
                save_dir: str,
                config: dict,
                bbox: list[float],
                cam_intrinsic:np.ndarray,
                ):

        super().__init__(
            args=args,
            target=target,
            planner=OccupancyGrid3DPathPlanner(config),
            save_dir=save_dir,
            cam_intrinsic=cam_intrinsic,
            bbox=bbox,
            get_rgbd=self.get_rgbd,
            get_cam_extrinsic=self.get_cam_extrinsic,
            move_fn=self.move_actions,
            rotate_fn=self.rotate_fn,
        )

    def move_actions(self):
        print(f'Suppose to be moving')

    def get_cam_extrinsic(self):
        print(f'Suppose to return camera extrinsic parameters')

    def get_rgbd(self):
        print(f'Suppose to return rgb and depth')

    def rotate_fn(self):
        print(f'Suppose to make the robot rotate')