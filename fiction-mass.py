import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation

from classes import Box, Point


class System:
    def __init__(self, box, points, fiction_point):
        self.box = box
        self.points = points
        self.fiction_point = fiction_point

    def render(self, ax):
        lines = self.box.render(ax)
        for point in self.points:
            lines.append(point.render(ax, "black"))
        lines.append(self.fiction_point.render(ax, "grey"))
        return lines

    def apply_control(self, v_box, omega_box, v_points, v_fiction_point, dt):
        self.box.apply_control(v_box, omega_box, dt)
        for (point, v_point) in zip(self.points, v_points):
            v_rot = (
                R.from_rotvec(omega_box * dt).as_matrix()
                @ (point.position - system.box.center)
                - (point.position - system.box.center)
            ) / dt
            point.apply_control(v_rot + v_point + v_box, dt)
        v_rot = (
            R.from_rotvec(omega_box * dt).as_matrix()
            @ (fiction_point.position - system.box.center)
            - (fiction_point.position - system.box.center)
        ) / dt
        self.fiction_point.apply_control(v_rot + v_fiction_point + v_box, dt)


#  equation from the article


def omega_linear_system(system, r, v):
    M = system.box.mass
    m = system.fiction_point.mass
    mu = m / (M + m)
    J = (
        system.box.inertia_tensor
        + sum(
            [
                point.mass * np.linalg.norm(point.position, ord=2)
                for point in system.points
            ]
        )
        * np.eye(3)
        - sum(
            [
                point.mass * (np.array([point.position]).T @ np.array([point.position]))
                for point in system.points
            ]
        )
        + M
        * mu
        * (np.linalg.norm(r, ord=2) * np.eye(3) - np.array([r]).T @ np.array([r]))
    )
    b = -M * mu * np.cross(r, v).reshape((3, 1))
    omega = solve(J, b)
    return np.array([row[0] for row in omega])


def update_system(phase_number, axis_for_point, center_for_point, system, dt, ax):
    ax.lines.clear()
    if phase_number <= 0:
        return [system.render(ax)]

    # don't forget to update trajectory params
    axis_for_point = system.box.matrix.T @ axis_for_point
    center_for_point = (
        system.box.matrix.T @ (center_for_point - system.box.center) + system.box.center
    )

    r_fiction_point = system.fiction_point.position
    axis = system.box.matrix @ axis_for_point
    v_fiction_point = (
        R.from_rotvec(axis * dt).as_matrix() @ (r_fiction_point - center_for_point)
        - (r_fiction_point - center_for_point)
    ) / dt
    v_points = np.resize(v_fiction_point, (len(system.points), 3))
    omega_box = omega_linear_system(system, r_fiction_point, v_fiction_point)
    v_box = -(
        sum(
            point.mass * (np.cross(omega_box, point.position) + v_fiction_point)
            for point in system.points
        )
        / (system.box.mass + system.fiction_point.mass)
    )
    system.apply_control(v_box, omega_box, v_points, v_fiction_point, dt)
    return [system.render(ax)]


if __name__ == "__main__":
    box_center = np.array([0.0, 0.0, 0.0])
    box_sizes = np.array([3.0, 5.0, 9.0])
    box_orientation = np.array([0.0, 0.0, 0.0])
    box_mass = 1000
    box_inertia_tensor = np.array(
        [[300.0, 0.0, 0.0], [0.0, 400.0, 0.0], [0.0, 0.0, 500.0]]
    )

    positions = np.array(
        [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-2.0, -2.0, -2.0]]
    )
    masses = np.array([10] * len(positions))
    points = np.array(
        [
            Point(position + box_center, mass)
            for (position, mass) in zip(positions, masses)
        ]
    )

    total_mass = sum(masses)
    fiction_point_position = (
        sum([mass * position for (position, mass) in zip(positions, masses)])
        / total_mass
    )

    fiction_point = Point(fiction_point_position + box_center, total_mass)

    cube = Box(box_center, box_sizes, box_orientation, box_mass, box_inertia_tensor)
    system = System(cube, points, fiction_point)

    axis_for_point = np.array([0, 1, 0])
    axis_for_point = axis_for_point / np.linalg.norm(axis_for_point)

    center_for_point = box_center + np.array(
        [1, 0, 0]
    )  # added array must be orthogonal to the vector axis_for_point

    n = 301
    dt = 2 * np.pi / (n - 1)
    phase = np.arange(0, n, 1)
    fps = 10

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

    limits = []
    for i in range(3):
        limits.append(
            np.array([box_center[i], box_center[i]])
            + np.array([-max(box_sizes), max(box_sizes)]) * 0.7
        )

    ax.set(xlim3d=limits[0], xlabel="X")
    ax.set(ylim3d=limits[1], ylabel="Y")
    ax.set(zlim3d=limits[2], zlabel="Z")

    animation = FuncAnimation(
        fig=fig,
        func=update_system,
        frames=phase,
        fargs=(axis_for_point, center_for_point, system, dt, ax),
        interval=1000 / fps,
        repeat=False,
    )
    fn = "cube_rotation_funcanimation"
    animation.save(fn + ".html", writer="html", fps=fps)
