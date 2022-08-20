import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation


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
        self.inertia_tensor = box.inertia_tensor
        for point in points[::-1]:
            r = np.array([point.position])
            m = point.mass
            self.inertia_tensor += m * np.linalg.norm(r) ** 2 * np.eye(3)
            self.inertia_tensor -= m * (r.T @ r)

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


#  equation from the article
def omega_linear_system(system, r, v):
    M = system.box.mass
    m = sum([point.mass for point in system.points[::-1]])
    mu = M * m / (M + m)
    r = np.array([r])
    J = system.inertia_tensor + mu * (np.linalg.norm(r)**2 * np.eye(3) - r.T @ r)
    b = -mu * np.cross(r, v).reshape((3, 1))
    omega = solve(J, b)
    return np.array([row[0] for row in omega])


def update_system(phase_number, axis_for_point, center_for_point, system, dt, ax):
    ax.lines.clear()
    if (phase_number > 0):
        r = system.points[-1].position
        axis = system.box.matrix @ axis_for_point
        v = (R.from_rotvec(axis * dt).as_matrix()
             @ (system.points[-1].position - center_for_point)
             - system.points[-1].position + center_for_point) / dt
#         v = np.cross(axis_for_point, center_for_point - r)
        omega_box = omega_linear_system(system, r, v)
        v_box = -(sum(point.mass * (np.cross(omega_box, point.position) + v)
                     for point in system.points[::-1])
                  / (system.box.mass + sum([point.mass for point in system.points[::-1]])))
        v_points = np.resize(v, (len(system.points), 3))
        system.apply_control(v_box, omega_box, v_points, dt)
    return [system.render(ax)]


if __name__ == "__main__":
    box_center = np.array([0.0, 0.0, 0.0])
    box_sizes = np.array([3.0, 5.0, 9.0])
    box_orientation = np.array([0.0, 0.0, 0.0])
    box_mass = 1000
    box_inertia_tensor = np.array([[300.0, 0, 0], [0, 400.0, 0], [0, 0, 500.0]])

    cube = Box(box_center, box_sizes, box_orientation, box_mass, box_inertia_tensor)

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

    system = System(cube, points)

    axis_for_point = np.array([0, 1, 0])
    axis_for_point = axis_for_point / np.linalg.norm(axis_for_point)
    center_for_point = np.array(
        [1, 0, 0]
    )  # вектор должен быть ортогонален axis_for_point

    n = 401
    dt = 2 * np.pi / (n - 1)
    phase = np.arange(0, n, 1)

    fps = 10

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
