import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


class Box:
    def __init__(self, center, sizes, orientation):
        self.center = center
        self.sizes = sizes
        self.roll = R.from_euler('xyz', np.array([orientation[0], 0, 0]))
        self.pitch = R.from_euler('xyz', np.array([0, orientation[1], 0]))
        self.yaw = R.from_euler('xyz', np.array([0, 0, orientation[2]]))
        self.matrix = self.yaw.as_matrix() @ self.pitch.as_matrix() @ self.roll.as_matrix()
    
    def box_edges(self):
        edges = []
        
        for j in (-0.5, 0.5):
            for k in (-0.5, 0.5):
                edges.append(np.array([[-self.sizes[0] / 2,
                                        j * self.sizes[1],
                                        k * self.sizes[2]],
                                       [self.sizes[0] / 2,
                                        j * self.sizes[1],
                                        k * self.sizes[2]]]).T)
        
        for k in (-0.5, 0.5):
            for i in (-0.5, 0.5):
                edges.append(np.array([[i * self.sizes[0],
                                        -self.sizes[1] / 2,
                                        k * self.sizes[2]],
                                       [i * self.sizes[0],
                                        self.sizes[1] / 2,
                                        k * self.sizes[2]]]).T)
        
        for i in (-0.5, 0.5):
            for j in (-0.5, 0.5):
                edges.append(np.array([[i * self.sizes[0],
                                        j * self.sizes[1],
                                        -self.sizes[2] / 2],
                                       [i * self.sizes[0],
                                        j * self.sizes[1],
                                        self.sizes[2] / 2]]).T)
        
        edges = np.array(edges)
        
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
        
        del colors
        del static_axes
        del vector


if __name__ == "__main__":
    box_center = np.array([0, 1, -5])
    box_sizes = np.array([2, 2, 2])
    
    box_orientation = -np.array([np.pi/2, np.pi/4, 0])
    
    cube = Box(box_center, box_sizes, box_orientation)
    
    ax = plt.figure(figsize=(7, 7)).add_subplot(projection='3d')
    
    cube.render(ax)
