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
        # center of the body
        ax.scatter(self.center[0], self.center[1], self.center[2], color="black", s=100)
        
        colors = ['red', 'blue', 'green']
        static_axes = np.eye(3)
        
        for i in range(3):            
            # fixed coordinate system
            vector = np.array([self.center, self.center + static_axes[i]]).T
            ax.plot(vector[0], vector[1], vector[2], color=colors[i])
            
            # coordinate system associated with the body
            vector = np.array([self.center, self.center + self.matrix[i]]).T
            ax.plot(vector[0], vector[1], vector[2], color=colors[i])
        
        # body
        for edge in self.box_edges():
            ax.plot(edge[0], edge[1], edge[2], color='grey')


def update_cube(phase_number, orientation, plot):
    return [Box(box_center, box_sizes, orientation[phase_number]).render(ax)]


if __name__ == "__main__":
    box_center = np.array([0, 1, -5])
    box_sizes = np.array([2, 2, 2])
    
    box_orientation = -np.array([np.pi/2, 7*np.pi/8, 0])
    
    cube = Box(box_center, box_sizes, box_orientation)
    
    fig = plt.figure()
    ax = plt.figure(figsize=(7, 7)).add_subplot(projection='3d')
    
    phase = np.arange(0, 3*np.pi/2, 0.05)
    orientation = []
    
    for p in np.arange(0, np.pi/2, 0.05):
        orientation.append(box_orientation + [p, 0, 0])
    
    box_orientation += [np.pi/2, 0, 0]
    
    for p in np.arange(0, np.pi/2, 0.05):
        orientation.append(box_orientation + [0, p, 0])
    
    box_orientation += [0, np.pi/2, 0]
    n = len(orientation)
    
    for i in range(len(phase) - n):
        orientation.append(box_orientation + [0, 0, i * 0.05])
    
    orientation = np.array(orientation)
    
    plot = [cube.render(ax)]
    fps = 10
    frn = len(phase)
    
    animation = FuncAnimation(fig=fig,
                              func=update_cube,
                              frames=frn,
                              fargs=(orientation,plot),
                              interval=1000/fps,
                              blit=True,
                              repeat=False)
    
    fn = 'cube_rotation_funcanimation'
    animation.save(fn+'.html',writer='html',fps=fps)
