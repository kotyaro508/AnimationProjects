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
        phi = np.linalg.norm(omega) * dt
        omega = omega / np.linalg.norm(omega)
        self.matrix = self.matrix @ np.array([[omega[0]**2 + np.cos(phi)*(1 - omega[0]**2),
                                 omega[0]*omega[1]*(1 - np.cos(phi)) + omega[2]*np.sin(phi),
                                 omega[0]*omega[2]*(1 - np.cos(phi)) - omega[1]*np.sin(phi)],
                                [omega[0]*omega[1]*(1 - np.cos(phi)) - omega[2]*np.sin(phi),
                                 omega[1]**2 + np.cos(phi)*(1 - omega[1]**2),
                                 omega[1]*omega[2]*(1 - np.cos(phi)) + omega[0]*np.sin(phi)],
                                [omega[0]*omega[2]*(1 - np.cos(phi)) + omega[1]*np.sin(phi),
                                 omega[1]*omega[2]*(1 - np.cos(phi)) - omega[0]*np.sin(phi),
                                 omega[2]**2 + np.cos(phi)*(1 - omega[2]**2)]])

# object update function
def update_cube(phase_number, cube, center, orientation, dt, ax):
    ax.clear()
    ax.set(xlim3d=limits[0], xlabel='X')
    ax.set(ylim3d=limits[1], ylabel='Y')
    ax.set(zlim3d=limits[2], zlabel='Z')
    cube.apply_control(center[phase_number], orientation[phase_number], dt)
    return [cube.render(ax)]


if __name__ == "__main__":
#     object creation
    box_center = np.array([0.0, 0.0, 0.0])
    box_sizes = np.array([1.0, 2.0, 4.0])
    box_orientation = np.array([0.0, 0.0, 0.0])    
    cube = Box(box_center, box_sizes, box_orientation)
    
#     figure creation
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection="3d")
    
#     fixing the scale of the figure axes
    limits = []
    for i in range(3):
        limits.append(np.array([box_center[i], box_center[i]]) +
                      np.array([-max(box_sizes), max(box_sizes)])*0.7)
    ax.set(xlim3d=limits[0], xlabel='X')
    ax.set(ylim3d=limits[1], ylabel='Y')
    ax.set(zlim3d=limits[2], zlabel='Z')
    
#     number of phases
    n = 300
    phase = np.arange(0, n, 1)
    
#     time step
    dt = 0.01
    
#     angular velocity
    orientation = []
    orientation.extend([[np.pi/2, 0, 0]] * 100)
    orientation.extend([[0, np.pi/2, 0]] * 100)
    orientation.extend([[0, 0, np.pi/2]] * 100)
    orientation = np.array(orientation)
    
#     no transfer
    center = np.zeros((n, 3))

#   plot = [cube.render(ax)]

#     elimination of incomprehensible initial shift by 2*dt
    cube.apply_control(np.array([0, 0, 0]),
                       -2*orientation[0],
                       dt)
    
#     animation creation
    fps = 10
    animation = FuncAnimation(fig=fig,
                              func=update_cube,
                              frames=n,
                              fargs=(cube,center,orientation,dt,ax),
                              interval=1000/fps,
                              repeat=False)
    
#     file saving
    fn = 'cube_rotation_funcanimation'
    plt.show()
    animation.save(fn+'.html',writer='html',fps=fps)
