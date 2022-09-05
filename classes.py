import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


class Box:
    def __init__(self, center, sizes, orientation, mass, inertia_tensor):
        self.center = center
        self.sizes = sizes
        self.matrix = R.from_euler("xyz", orientation).as_matrix()
        self.mass = mass
        self.inertia_tensor = inertia_tensor

    def box_edges(self):
        edges = []

        for j in (-1, 1):
            for k in (-1, 1):
                edges.append(
                    np.array(
                        [
                            [-self.sizes[0], j * self.sizes[1], k * self.sizes[2]],
                            [self.sizes[0], j * self.sizes[1], k * self.sizes[2]],
                        ]
                    ).T
                )

        for k in (-1, 1):
            for i in (-1, 1):
                edges.append(
                    np.array(
                        [
                            [i * self.sizes[0], -self.sizes[1], k * self.sizes[2]],
                            [i * self.sizes[0], self.sizes[1], k * self.sizes[2]],
                        ]
                    ).T
                )

        for i in (-1, 1):
            for j in (-1, 1):
                edges.append(
                    np.array(
                        [
                            [i * self.sizes[0], j * self.sizes[1], -self.sizes[2]],
                            [i * self.sizes[0], j * self.sizes[1], self.sizes[2]],
                        ]
                    ).T
                )

        edges = np.array(edges) / 2

        for i in range(12):
            edges[i] = self.matrix.T @ edges[i] + np.array([self.center, self.center]).T

        return edges

    def render(self, ax):
        # center of the box

        colors = ["red", "blue", "green"]
        static_axes = np.eye(3)

        lines = []

        for i in range(3):
            # fixed coordinate system
            vector = np.array([self.center, self.center + static_axes[i]]).T
            lines.append(ax.plot(vector[0], vector[1], vector[2], color=colors[i])[0])

            # coordinate system associated with the box
            vector = np.array([self.center, self.center + self.matrix[i]]).T
            lines.append(ax.plot(vector[0], vector[1], vector[2], color=colors[i])[0])

        # box
        for edge in self.box_edges():
            lines.append(ax.plot(edge[0], edge[1], edge[2], color="grey")[0])

        return lines

    def apply_control(self, v, omega, dt):
        self.center = self.center + v * dt
        self.matrix = self.matrix @ R.from_rotvec(omega * dt).as_matrix().T


class Point:
    def __init__(self, position, mass):
        self.position = position
        self.mass = mass

    def render(self, ax, color):
        return ax.plot(
            [self.position[0]],
            [self.position[1]],
            [self.position[2]],
            color=color,
            marker="o",
            markersize=5,
        )

    def apply_control(self, v, dt):
        self.position = self.position + v * dt


class System:
    def __init__(self, box, points):
        self.box = box
        self.points = points

    def render(self, ax):
        lines = self.box.render(ax)
        for point in self.points[:-1]:
            lines.append(point.render(ax, "black"))
        lines.append(self.points[-1].render(ax, "grey"))
        return lines

    def apply_control(self, v_box, omega_box, v_points, dt):
        self.box.apply_control(v_box, omega_box, dt)
        for (point, v_point) in zip(self.points, v_points):
            v = (
                R.from_rotvec(omega_box * dt).as_matrix() @ point.position
                - point.position
            ) / dt
            point.apply_control(v + v_point + v_box, dt)
