import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation


class Box:
    def __init__(self, center, sizes, orientation):
        self.center = center
        self.sizes = sizes
        self.matrix = R.from_euler('xyz', orientation).as_matrix().T

    def box_edges(self):
        edges = []

        for j in (-1, 1):
            for k in (-1, 1):
                edges.append(np.array([[-self.sizes[0],
                                        j * self.sizes[1],
                                        k * self.sizes[2]],
                                       [self.sizes[0],
                                        j * self.sizes[1],
                                        k * self.sizes[2]]]).T)

        for k in (-1, 1):
            for i in (-1, 1):
                edges.append(np.array([[i * self.sizes[0],
                                        -self.sizes[1],
                                        k * self.sizes[2]],
                                       [i * self.sizes[0],
                                        self.sizes[1],
                                        k * self.sizes[2]]]).T)

        for i in (-1, 1):
            for j in (-1, 1):
                edges.append(np.array([[i * self.sizes[0],
                                        j * self.sizes[1],
                                        -self.sizes[2]],
                                       [i * self.sizes[0],
                                        j * self.sizes[1],
                                        self.sizes[2]]]).T)

        edges = np.array(edges) / 2

        for i in range(12):
            edges[i] = self.matrix.T @ edges[i] + np.array([self.center, self.center]).T

        return edges

    def render(self, ax):
        # center of the box

        colors = ['red', 'blue', 'green']
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
            lines.append(ax.plot(edge[0], edge[1], edge[2], color='grey')[0])

        return lines

    def apply_control(self, v, omega, dt):
        self.center = self.center + v * dt
        self.matrix = self.matrix @ R.from_rotvec(omega * dt).as_matrix().T


class Point:
    def __init__(self, position):
        self.position = position

    def render(self, ax):
        return ax.plot([self.position[0]], [self.position[1]],
                       [self.position[2]], color='black',
                       marker="o", markersize=5)

    def apply_control(self, is_linear, velocity, dt):
        if is_linear:
            # velocity is linear
            self.position = self.position + velocity * dt
        else:
            # velocity is angular velocity of the box
            self.position = R.from_rotvec(velocity * dt).as_matrix() @ self.position


class System:
    def __init__(self, box, points):
        self.box = box
        self.points = points

    def render(self, ax):
        lines = self.box.render(ax)
        for point in self.points:
            lines.append(point.render(ax))
        return lines

    def apply_control(self, v_box, omega_box, v_points, dt):
        self.box.apply_control(v_box, omega_box, dt)
        for (point, v_point) in zip(self.points, v_points):
            if np.linalg.norm(v_point):
                # point with its own motion
                point.apply_control(True, v_point, dt)
            else:
                # point without its own motion - rotates with the box
                point.apply_control(False, omega_box, dt)


def update_system(phase_number, system, dt, ax):
    ax.lines.clear()
    v_box = np.zeros(3)
    v_points = np.zeros((len(system.points), 3))
    omega_box = np.zeros(3)
    if 0 < phase_number <= 100:
        omega_box = np.array([np.pi/2, 0, 0])
    elif 100 < phase_number <= 200:
        omega_box = np.array([0, np.pi/2, 0])
    elif 200 < phase_number <= 300:
        omega_box = np.array([0, 0, np.pi/2])
    system.apply_control(v_box, omega_box, v_points, dt)
    return [system.render(ax)]


if __name__ == "__main__":
    box_center = np.array([0.0, 0.0, 0.0])
    box_sizes = np.array([3.0, 5.0, 9.0])
    box_orientation = np.array([0.0, 0.0, 0.0])

    cube = Box(box_center, box_sizes, box_orientation)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

    limits = []
    for i in range(3):
        limits.append(np.array([box_center[i], box_center[i]]) +
                      np.array([-max(box_sizes), max(box_sizes)]) * 0.7)

    ax.set(xlim3d=limits[0], xlabel='X')
    ax.set(ylim3d=limits[1], ylabel='Y')
    ax.set(zlim3d=limits[2], zlabel='Z')

    points = np.array([Point(point + box_center)
                       for point in np.array([[1.0, 0.0, 0.0],
                                              [-1.0, 0.0, 0.0],
                                              [0.0, 2.0, 0.0],
                                              [0.0, -2.0, 0.0],
                                              [0.0, 0.0, 4.0],
                                              [0.0, 0.0, -4.0]])])

    system = System(cube, points)

    n = 301
    dt = 0.01
    phase = np.arange(0, n, 1)

    fps = 10

    animation = FuncAnimation(fig=fig,
                              func=update_system,
                              frames=phase,
                              fargs=(system, dt, ax),
                              interval=1000 / fps,
                              repeat=False)
    fn = 'cube_rotation_funcanimation'
    animation.save(fn + '.html', writer='html', fps=fps)
