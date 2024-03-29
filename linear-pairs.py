import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation, FFMpegWriter

from classes import Box, Point, System
from Runge-Kutta import rk4


def func(t, x, alpha):
    f = np.zeros(3)
    f[0] = (
        -alpha[2] * (b[2] + a[0] ** 2 + a[1] ** 2)
        + alpha[0] * a[2] * x[2]
        + alpha[1] * a[1] * x[1]
        - alpha[2] * (x[2] ** 2 + x[0] ** 2)
    ) / a[0]
    f[1] = (
        -alpha[0] * (b[0] + a[1] ** 2 + a[2] ** 2)
        + alpha[1] * a[0] * x[0]
        + alpha[2] * a[2] * x[2]
        - alpha[0] * (x[0] ** 2 + x[1] ** 2)
    ) / a[1]
    f[2] = (
        -alpha[1] * (b[1] + a[2] ** 2 + a[0] ** 2)
        + alpha[2] * a[1] * x[1]
        + alpha[0] * a[0] * x[0]
        - alpha[1] * (x[1] ** 2 + x[2] ** 2)
    ) / a[2]
    return f


def update_system(phase_number, system, X, alpha, dt, ax):
    ax.lines.clear()
    v_box = np.zeros(3)
    v_points = np.resize(v_box, (len(system.points), 3))
    if phase_number > 0:
        system.apply_control(v_box, alpha, v_points, dt)
        axes = system.box.matrix.T
        v_points[0] = (
            axes @ np.array([0, X[0][phase_number] - X[0][phase_number - 1], 0]) / dt
        )
        v_points[1] = -v_points[0]
        v_points[2] = (
            axes @ np.array([0, 0, X[1][phase_number] - X[1][phase_number - 1]]) / dt
        )
        v_points[3] = -v_points[2]
        v_points[4] = (
            axes @ np.array([X[2][phase_number] - X[2][phase_number - 1], 0, 0]) / dt
        )
        v_points[5] = -v_points[4]
        system.apply_control(np.zeros(3), np.zeros(3), v_points, dt)
    return [system.render(ax)]


if __name__ == "__main__":
    box_center = np.array([0.0, 0.0, 0.0])
    box_sizes = np.array([3.0, 3.0, 3.0])
    box_orientation = np.array([0.0, 0.0, 0.0])
    box_mass = 1000
    box_inertia_tensor = np.array([[300, 0, 0], [0, 400, 0], [0, 0, 500]])

    positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
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

    b = sum(system.box.inertia_tensor) / (2 * system.points[0].mass)
    a = np.array([np.linalg.norm(point.position) for point in system.points[::2]])
    alpha = np.array([1.0, 1.0, 1.0])
    alpha = alpha / np.linalg.norm(alpha)

    phi_span = np.array([0.0, 2 * np.pi])
    x0 = np.array([0.0, 0.0, 0.0])
    n = 8001
    dt = 2 * np.pi / (n - 1)
    X = rk4(phi_span, x0, n, alpha)

    for i in range(n):
        if min(abs(X[:, i])) > 2:
            n = i + 1
            break
    X = (X.T[:n]).T
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
        fargs=(system, X, alpha, dt, ax),
        interval=1000 / fps,
        repeat=False,
    )

    FFwriter = FFMpegWriter(fps=fps, extra_args=["-vcodec", "libx264"])
    fn = "cube_rotation_funcanimation"
    animation.save(fn + ".mp4", writer=FFwriter)
