import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation

from classes import Box, Point, System


#  equation from the article
def omega_linear_system(system, r, v):
    M = system.box.mass
    m = sum([point.mass for point in system.points[:-1]])
    mu = M * m / (M + m)
    J = (
        system.box.inertia_tensor
        + sum(
            [
                point.mass * np.linalg.norm(point.position, ord=2)
                for point in system.points[:-1]
            ]
        )
        * np.eye(3)
        - sum(
            [
                point.mass * (np.array([point.position]).T @ np.array([point.position]))
                for point in system.points[:-1]
            ]
        )
        + mu * (np.linalg.norm(r, ord=2) * np.eye(3) - np.array([r]).T @ np.array([r]))
    )
    b = -mu * np.cross(r, v).reshape((3, 1))
    omega = solve(J, b)
    return np.array([row[0] for row in omega])


def update_system(phase_number, axis_for_point, center_for_point, system, dt, ax):
    ax.lines.clear()
    #     v_box = np.zeros(3)
    #     v_points = np.zeros((len(system.points), 3))
    if phase_number > 0:
        r = system.points[-1].position
        axis = system.box.matrix @ axis_for_point
        v = (
            R.from_rotvec(axis * dt).as_matrix()
            @ (system.points[-1].position - center_for_point)
            - system.points[-1].position
            + center_for_point
        ) / dt
        #         v = np.cross(axis_for_point, center_for_point - r)
        omega_box = omega_linear_system(system, r, v)
        v_box = -(
            sum(
                point.mass * (np.cross(omega_box, point.position) + v)
                for point in system.points[:-1]
            )
            / (system.box.mass + sum([point.mass for point in system.points[::-1]]))
        )
        v_points = np.resize(v, (len(system.points), 3))
        system.apply_control(v_box, omega_box, v_points, dt)
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
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [-2.0, -2.0, -2.0],
            [0.0, 0.0, 0.0],
        ]
    )
    points_masses = np.array([10] * 6)
    points = np.array(
        [
            Point(point + box_center, mass)
            for (point, mass) in zip(positions, points_masses)
        ]
    )

    cube = Box(box_center, box_sizes, box_orientation, box_mass, box_inertia_tensor)
    system = System(cube, points)

    axis_for_point = np.array([0, 1, 0])
    axis_for_point = axis_for_point / np.linalg.norm(axis_for_point)
    center_for_point = np.array(
        [1, 0, 0]
    )  # вектор должен быть ортогонален axis_for_point

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
