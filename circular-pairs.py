import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation, FFMpegWriter


class Box:
    def __init__(self, center, sizes, orientation, mass, inertia_tensor):
        self.center = center
        self.sizes = sizes
        self.matrix = R.from_euler('xyz', orientation).as_matrix()
        self.mass = mass
        self.inertia_tensor = inertia_tensor

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

        edges = np.array(edges)/2

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
    def __init__(self, position, mass):
        self.position = position
        self.mass = mass

    def render(self, ax, color):
        return ax.plot([self.position[0]], [self.position[1]],
                        [self.position[2]], color=color,
                        marker="o", markersize=5)

    def apply_control(self, v, dt):
        self.position = self.position + v * dt


class System:
    def __init__(self, box, points):
        self.box = box
        self.points = points

    def render(self, ax):
        lines = self.box.render(ax)
        for point in self.points:
            lines.append(point.render(ax, 'black'))
        return lines

    def apply_control(self, v_box, omega_box, v_points, dt):
        self.box.apply_control(v_box, omega_box, dt)
        for (point, v_point) in zip(self.points, v_points):
            v = (R.from_rotvec(omega_box * dt).as_matrix() @ point.position
                 - point.position) / dt
            point.apply_control(v + v_point + v_box, dt)


def func(t, x, alpha):
    f = np.zeros(3)
    f[0] = -((a[2]/a[0])**2 * (alpha[2] * np.sin(x[2])**2 - alpha[0] * np.sin(x[2]) * np.cos(x[2])) +
             (a[1]/a[0])**2 * (alpha[2] * np.cos(x[1])**2 - alpha[1] * np.sin(x[1]) * np.cos(x[1])) +
             alpha[2] * (1 + b[2]/(a[0]**2)))
    f[1] = -((a[0]/a[1])**2 * (alpha[0] * np.sin(x[0])**2 - alpha[1] * np.sin(x[0]) * np.cos(x[0])) +
             (a[2]/a[1])**2 * (alpha[0] * np.cos(x[2])**2 - alpha[2] * np.sin(x[2]) * np.cos(x[2])) +
             alpha[0] * (1 + b[0]/(a[1]**2)))
    f[2] = -((a[1]/a[2])**2 * (alpha[1] * np.sin(x[1])**2 - alpha[2] * np.sin(x[1]) * np.cos(x[1])) +
             (a[0]/a[2])**2 * (alpha[1] * np.cos(x[0])**2 - alpha[0] * np.sin(x[0]) * np.cos(x[0])) +
             alpha[1] * (1 + b[1]/(a[2]**2)))
    return f


def rk4(phi_span, x0, npoints, alpha):
    T = np.linspace(phi_span[0], phi_span[1], npoints + 1)
    h = T[1] - T[0]
    X = np.empty((len(x0), npoints + 1))
    X[:, 0] = x0.copy()
    for k in range(npoints):
        k1 = func(T[k], X[:, k], alpha)
        k2 = func(T[k] + h/2, X[:, k] + h*k1/2, alpha)
        k3 = func(T[k] + h/2, X[:, k] + h*k2/2, alpha)
        k4 = func(T[k] + h, X[:, k] + h*k3, alpha)
        X[:, k+1] = X[:, k] + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return X


def update_system(phase_number, system, X, alpha, dt, ax):
    ax.lines.clear()
    v_box = np.zeros(3)
    v_points = np.zeros((len(system.points), 3))
    if (phase_number > 0):
        system.apply_control(v_box, alpha, v_points, dt)
        axes = system.box.matrix
        v_points[0] = (R.from_rotvec(axes[2] * (X[0][phase_number]
                                                - X[0][phase_number - 1]) / (2 * np.pi)).as_matrix()
                       @ system.points[0].position
                       - system.points[0].position) / dt
        v_points[1] = -v_points[0]
        v_points[2] = (R.from_rotvec(axes[0] * (X[1][phase_number]
                                                - X[1][phase_number - 1]) / (2 * np.pi)).as_matrix()
                       @ system.points[2].position
                       - system.points[2].position) / dt
        v_points[3] = -v_points[2]
        v_points[4] = (R.from_rotvec(axes[1] * (X[2][phase_number]
                                                - X[2][phase_number - 1]) / (2 * np.pi)).as_matrix()
                       @ system.points[4].position
                       - system.points[4].position) / dt
        v_points[5] = -v_points[4]
        system.apply_control(np.zeros(3), np.zeros(3), v_points, dt)
    return [system.render(ax)]


if __name__ == "__main__":
    box_center = np.array([0.0, 0.0, 0.0])
    box_sizes = np.array([3.0, 3.0, 3.0])
    box_orientation = np.array([0.0, 0.0, 0.0])
    box_mass = 1000
    box_inertia_tensor = np.array([[300, 0, 0],
                                   [0, 400, 0],
                                   [0, 0, 500]])
    
    cube = Box(box_center, box_sizes, box_orientation, box_mass, box_inertia_tensor)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    
    limits = []
    for i in range(3):
        limits.append(np.array([box_center[i], box_center[i]]) +
                      np.array([-max(box_sizes), max(box_sizes)])*0.7)
    
    ax.set(xlim3d=limits[0], xlabel='X')
    ax.set(ylim3d=limits[1], ylabel='Y')
    ax.set(zlim3d=limits[2], zlabel='Z')
    
    positions = np.array([[1.0, 0.0, 0.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, -1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [0.0, 0.0, -1.0]])
    points_masses = np.array([10] * 6)
    points = np.array([Point(point + box_center, mass)
                       for (point, mass) in zip(positions, points_masses)])
    
    system = System(cube, points)
    
    b = sum(system.box.inertia_tensor) / (2 * system.points[0].mass)
    a = np.array([np.linalg.norm(point.position) for point in system.points[::2]])
    alpha = np.array([4.0, 2.0, 1.0])
    alpha = alpha / np.linalg.norm(alpha)
    
    phi_span = np.array([0.0, 2 * np.pi])
    x0 = np.array([0.0, 0.0, 0.0])
    n = 801
    dt = (phi_span[1] - phi_span[0]) / (n - 1)
    X = rk4(phi_span, x0, n, alpha)
    
    phase = np.arange(0, n, 1)
    
    fps = 10
    
    animation = FuncAnimation(fig=fig,
                              func=update_system,
                              frames=phase,
                              fargs=(system,X,alpha,dt,ax),
                              interval=1000/fps,
                              repeat=False)
    
    FFwriter = FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264'])
    fn = 'cube_rotation_funcanimation'
    animation.save(fn+'.wmv',writer=FFwriter)
