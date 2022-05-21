import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation


class Box:
    def __init__(self, center, sizes, orientation):
        self.center = center
        self.sizes = sizes
        self.matrix = R.from_euler('xyz', orientation).as_matrix()
    
    def box_edges(self):
        edges = []
        
        for j in (-1, 1):
            for k in (-1, 1):
                edges.append(np.array([[-self.sizes[0],
                                        j * self.sizes[1],
                                        k * self.sizes[2]],
                                       [self.sizes[0],
                                        j * self.sizes[1],
                                        k * self.sizes[2]]]).T / 2)
        
        for k in (-1, 1):
            for i in (-1, 1):
                edges.append(np.array([[i * self.sizes[0],
                                        -self.sizes[1],
                                        k * self.sizes[2]],
                                       [i * self.sizes[0],
                                        self.sizes[1],
                                        k * self.sizes[2]]]).T / 2)
        
        for i in (-1, 1):
            for j in (-1, 1):
                edges.append(np.array([[i * self.sizes[0],
                                        j * self.sizes[1],
                                        -self.sizes[2]],
                                       [i * self.sizes[0],
                                        j * self.sizes[1],
                                        self.sizes[2]]]).T / 2)
        
        edges = np.array(edges)
        
        for i in range(12):
            edges[i] = self.matrix.T @ edges[i] + np.array([self.center, self.center]).T
        
        return edges
                    
    def render(self, ax):
        colors = ['red', 'blue', 'green']
        static_axes = np.eye(3)
        

        lines = []
        for i in range(3):            
            # fixed coordinate system
            vector = np.array([self.center, self.center + static_axes[i]]).T
            lines.append(ax.plot(vector[0], vector[1], vector[2], color=colors[i])[0])
            
            # coordinate system associated with the body
            vector = np.array([self.center, self.center + self.matrix[i]]).T
            lines.append(ax.plot(vector[0], vector[1], vector[2], color=colors[i])[0])
        
        # body
        for edge in self.box_edges():
            lines.append(ax.plot(edge[0], edge[1], edge[2], color='grey')[0])

        return lines

    def apply_control(self, v, omega, dt):
        self.center = self.center + v * dt
        self.matrix = self.matrix @ R.from_rotvec(omega * dt).as_matrix().T


class Point:
    def __init__(self, location):
        self.location = location

    def render(self, ax, color='black', size=5):
        return [ax.plot([self.location[0]], [self.location[1]],
                        [self.location[2]], color=color,
                        marker="o", markersize=size)]

    def apply_control(self, v, dt):
        self.location = self.location + v * dt


def update_cube(phase_number, cube, dt, ax):
    ax.lines.clear()
    linear_velocity = np.zeros(3)
    i = float(0 < phase_number <= 100)
    j = float(100 < phase_number <= 200)
    k = float(200 < phase_number <= 300)
    angular_velocity = np.array([i, j, k]) * np.pi/2
    cube.apply_control(linear_velocity, angular_velocity, dt)
    return [cube.render(ax)]


if __name__ == "__main__":
    box_center = np.array([0.0, 0.0, 0.0])
    box_sizes = np.array([1.0, 2.0, 4.0])
    box_orientation = np.array([0.0, 0.0, 0.0])
    
    cube = Box(box_center, box_sizes, box_orientation)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

    limits = []

    for i in range(3):
        limits.append(np.array([box_center[i], box_center[i]]) +
                      np.array([-max(box_sizes), max(box_sizes)])*0.7)

    ax.set(xlim3d=limits[0], xlabel='X')
    ax.set(ylim3d=limits[1], ylabel='Y')
    ax.set(zlim3d=limits[2], zlabel='Z')

    n = 301
    dt = 0.01
    phase = np.arange(0, n, 1)

    fps = 10

    animation = FuncAnimation(fig=fig,
                              func=update_cube,
                              frames=phase,
                              fargs=(cube,dt,ax),
                              interval=1000/fps,
                              repeat=False)
    fn = 'cube_rotation_funcanimation'
    animation.save(fn+'.html',writer='html',fps=fps)
