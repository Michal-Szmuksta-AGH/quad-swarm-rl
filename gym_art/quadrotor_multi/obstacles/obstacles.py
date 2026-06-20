import copy
import numpy as np

from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs, collision_detection
from gym_art.quadrotor_multi.sensors.multiranger import get_multiranger_obs


class MultiObstacles:
    def __init__(self, obstacle_size=1.0, quad_radius=0.046, sensor_range=100.0,
                 obs_type='octomap', room_dims=(10., 10., 10.),
                 multiranger_max_range=4.0, multiranger_noise_std=0.0,
                 multiranger_fov_deg=27.0, multiranger_num_rays=8):
        self.size = obstacle_size
        self.obstacle_radius = obstacle_size / 2.0
        self.quad_radius = quad_radius
        self.pos_arr = []
        self.resolution = 0.1
        self.sensor_range = sensor_range
        self.obs_type = obs_type
        self.room_dims_xy = np.array([room_dims[0], room_dims[1]], dtype=np.float32)
        self.multiranger_max_range = multiranger_max_range
        self.multiranger_noise_std = multiranger_noise_std
        self.multiranger_fov_rad = float(multiranger_fov_deg) * np.pi / 180.0
        self.multiranger_num_rays = int(multiranger_num_rays)

    def _compute_obstacle_obs(self, quads_pos, quads_rot=None):
        if self.obs_type == 'octomap':
            quads_sdf_obs = 100 * np.ones((len(quads_pos), 9))
            return get_surround_sdfs(
                quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                resolution=self.resolution, sensor_range=self.sensor_range)
        if self.obs_type == 'multiranger':
            if quads_rot is None:
                raise ValueError("multiranger obs_type requires quads_rot (rotation matrices)")
            # Extract yaw from rotation matrix: yaw = atan2(R[1,0], R[0,0])
            quad_yaws = np.arctan2(quads_rot[:, 1, 0], quads_rot[:, 0, 0]).astype(np.float32)
            return get_multiranger_obs(
                quad_poses_xy=quads_pos[:, :2].astype(np.float32),
                quad_yaws=quad_yaws,
                obst_poses_xy=self.pos_arr[:, :2].astype(np.float32),
                obst_radius=float(self.obstacle_radius),
                room_dims_xy=self.room_dims_xy,
                max_range=float(self.multiranger_max_range),
                noise_std=float(self.multiranger_noise_std),
                fov_rad=self.multiranger_fov_rad,
                num_rays=self.multiranger_num_rays)
        raise ValueError(f"Unknown obs_type: {self.obs_type}")

    def reset(self, obs, quads_pos, pos_arr, quads_rot=None):
        self.pos_arr = copy.deepcopy(np.array(pos_arr))
        obstacle_obs = self._compute_obstacle_obs(quads_pos, quads_rot)
        obs = np.concatenate((obs, obstacle_obs), axis=1)
        return obs

    def step(self, obs, quads_pos, quads_rot=None):
        obstacle_obs = self._compute_obstacle_obs(quads_pos, quads_rot)
        obs = np.concatenate((obs, obstacle_obs), axis=1)
        return obs

    def collision_detection(self, pos_quads):
        quad_collisions = collision_detection(quad_poses=pos_quads[:, :2], obst_poses=self.pos_arr[:, :2],
                                              obst_radius=self.obstacle_radius, quad_radius=self.quad_radius)

        collided_quads_id = np.where(quad_collisions > -1)[0]
        collided_obstacles_id = quad_collisions[collided_quads_id]
        quad_obst_pair = {}
        for i, key in enumerate(collided_quads_id):
            quad_obst_pair[key] = int(collided_obstacles_id[i])

        return collided_quads_id, quad_obst_pair
