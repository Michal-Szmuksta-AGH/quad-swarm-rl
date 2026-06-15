"""
Interactive waypoint scenario.

The swarm flies to a user-controlled sequence of waypoints. In the render
window, press ENTER to advance to the next waypoint, R to reset to the first.
Edit DEFAULT_WAYPOINTS below to customise the route.

Requires --quads_use_obstacles=True (extends Scenario_o_base). Suggested:
  --quads_mode=o_waypoints --quads_use_obstacles=True --quads_view_mode global
  --quads_episode_duration=600   (long episode so episode resets don't interrupt)
"""

import copy

import numpy as np

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


# Edytuj listę pod własną trasę. Z = wysokość (m). Sensowne dla pokoju ~16x16x8.
DEFAULT_WAYPOINTS = [
    np.array([0.0, 0.0, 2.0]),
    np.array([5.0, 0.0, 2.0]),
    np.array([5.0, 5.0, 2.5]),
    np.array([-5.0, 5.0, 2.5]),
    np.array([-5.0, -5.0, 2.0]),
    np.array([5.0, -5.0, 2.0]),
    np.array([0.0, 0.0, 4.0]),
]


class Scenario_o_waypoints(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.waypoints = [np.array(w, dtype=float) for w in DEFAULT_WAYPOINTS]
        self.current_idx = 0
        self.approch_goal_metric = 1.0

    def _current_waypoint(self):
        return self.waypoints[self.current_idx]

    def _apply_current_waypoint(self):
        wp = self._current_waypoint()
        self.goals = np.array([wp.copy() for _ in range(self.num_agents)])
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

    def advance(self):
        self.current_idx = (self.current_idx + 1) % len(self.waypoints)
        print(f"[waypoints] -> #{self.current_idx} {self._current_waypoint().tolist()}")
        self._apply_current_waypoint()

    def reset_to_first(self):
        self.current_idx = 0
        print(f"[waypoints] reset -> #0 {self._current_waypoint().tolist()}")
        self._apply_current_waypoint()

    def step(self):
        # Cel zmienia się tylko na żądanie (ENTER / R), nie z upływem czasu.
        return

    def reset(self, obst_map=None, cell_centers=None):
        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None or cell_centers is None:
            raise NotImplementedError(
                "o_waypoints requires obstacles enabled (--quads_use_obstacles=True)"
            )

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        self.start_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        self.spawn_points = copy.deepcopy(self.start_point)

        self.update_formation_and_relate_param()

        self.current_idx = 0
        wp = self._current_waypoint()
        self.goals = np.array([wp.copy() for _ in range(self.num_agents)])
