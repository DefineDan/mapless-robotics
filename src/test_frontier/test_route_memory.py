import numpy as np
import quaternion
import unittest

from mapless_robotics.exploration.route_memory import RecordedRoute
from mapless_robotics.navigate.lotis_route_policy import chunk_route_ids, compact_route_ids


class SensorState:
    def __init__(self, yaw):
        self.rotation = quaternion.from_rotation_vector(np.array([0.0, yaw, 0.0]))


class State:
    def __init__(self, position, yaw=0.0):
        self.position = np.asarray(position, dtype=float)
        self.rotation = quaternion.one
        self.sensor_states = {'rgb': SensorState(yaw)}


def add(route, position, yaw=0.0):
    return route.append(np.zeros((2, 2, 3), dtype=np.uint8), State(position, yaw))


class RecordedRouteTests(unittest.TestCase):
    def test_near_views_create_weighted_edges_but_far_views_do_not(self):
        route = RecordedRoute(connect_distance=0.5)
        first = add(route, [0.0, 0.0, 0.0])
        nearby = add(route, [0.4, 0.0, 0.0])
        far = add(route, [2.0, 0.0, 0.0])

        self.assertIn(nearby, route.adjacency[first])
        self.assertAlmostEqual(route.adjacency[first][nearby]['distance'], 0.4)
        self.assertNotIn(far, route.adjacency[first])
        self.assertIn(far, route.adjacency[nearby])

    def test_shortest_path_uses_camera_angle_cost(self):
        route = RecordedRoute(connect_distance=1.1, heading_weight=0.5)
        start = add(route, [0.0, 0.0, 0.0], 0.0)
        target = add(route, [1.0, 0.0, 0.0], np.pi)
        detour_a = add(route, [0.0, 0.0, 1.0], 0.0)
        detour_b = add(route, [0.5, 0.0, 1.0], np.pi / 2)
        detour_c = add(route, [1.0, 0.0, 1.0], np.pi)

        self.assertEqual(route.shortest_path(start, target),
                         [start, detour_a, detour_b, detour_c, target])

    def test_close_loop_keeps_graph_branches_available(self):
        route = RecordedRoute(connect_distance=0.5)
        start = add(route, [0.0, 0.0, 0.0])
        branch = add(route, [0.4, 0.0, 0.0])
        end = add(route, [0.8, 0.0, 0.0])

        route.close_loop(branch, 'test')

        self.assertEqual(route.active, [start, branch])
        self.assertEqual(route.shortest_path(start, end), [start, branch, end])
        self.assertTrue(route.to_dict()['route_edges'])

    def test_compaction_keeps_camera_turns(self):
        states = [State([0.0, 0.0, 0.0], yaw) for yaw in (0.0, np.deg2rad(20), np.deg2rad(40))]

        self.assertEqual(compact_route_ids([0, 1, 2], states), [0, 1, 2])

    def test_chunks_overlap_without_exceeding_encoder_capacity(self):
        ids = list(range(12))
        states = [State([float(index), 0.0, 0.0]) for index in ids]

        chunks = chunk_route_ids(ids, states, max_frames=5, overlap=2)

        self.assertTrue(all(2 <= len(chunk) <= 5 for chunk in chunks))
        self.assertEqual(chunks[0][-2:], chunks[1][:2])
        self.assertEqual(chunks[1][-2:], chunks[2][:2])
        self.assertEqual(chunks[-1][-1], ids[-1])

    def test_sharp_view_change_starts_new_chunk_at_transition(self):
        ids = list(range(8))
        states = [State([float(index), 0.0, 0.0], 0.0 if index < 4 else np.pi)
                  for index in ids]

        chunks = chunk_route_ids(ids, states, max_frames=6, overlap=2)

        self.assertEqual(chunks, [[0, 1, 2, 3], [4, 5, 6, 7]])


if __name__ == '__main__':
    unittest.main()