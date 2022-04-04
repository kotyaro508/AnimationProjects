import matplotlib.pyplot as plt
import numpy as np


# the angle between the vectors u and v with the right to rotate around the z axis
def angle_between(u, v, z):
    u_norm = u / np.linalg.norm(u)
    v_norm = v / np.linalg.norm(v)
    cross = np.cross(u_norm, v_norm)
    if np.linalg.norm(cross) == 0:
        return np.arccos(np.dot(u_norm, v_norm))
    return (np.arccos(np.dot(u_norm, v_norm)) *
            np.dot(z / np.linalg.norm(z), cross / np.linalg.norm(cross))) % (2 * np.pi)


class Box:
    def __init__(self, center, sizes, x_axis, y_axis):
        self.center = center
        self.axes = np.array([x_axis, y_axis, np.cross(x_axis, y_axis)])
        self.sizes = sizes
        
        # knot line vector == intersection of two Oxy planes == cross product of two z axes
        knot_line_axis = (np.cross(np.array([0, 0, 1]), self.axes[2]) /
                          np.linalg.norm(np.cross(np.array([0, 0, 1]),
                                                  self.axes[2])))
        
        # one of two precession angles (in the set [0; pi) or in the set (pi; 2pi) )
        precession = angle_between(np.array([1, 0, 0]),
                                   knot_line_axis,
                                   np.array([0, 0, 1]))
        
        self.precession = precession % np.pi                               # precession angle (in the set [0; pi) )
        
        # set the knot line vector at an acute angle of precession to the fixed x-axis (multiply by 1 or -1)
        knot_line_axis *= (-2) * (precession // np.pi - 0.5)
        
        del precession
        
        self.nutation = angle_between(np.array([0, 0, 1]),                 # nutation angle (in the set [0; 2pi) )
                                      self.axes[2],
                                      knot_line_axis)
        
        self.own_rotation = angle_between(knot_line_axis,                  # own rotation angle
                                          self.axes[0],                     # (in the set [0; 2pi) )
                                          self.axes[2])
        
        del knot_line_axis
    
    def box_edges(self):
        edges = []
        
        for i in range(3):
            for j in (-0.5, 0.5):
                for k in (-0.5, 0.5):
                    
                    edges.append(np.array([self.center +
                                                j * self.sizes[i - 2] * self.axes[i - 2] +
                                                k * self.sizes[i - 1] * self.axes[i - 1] -
                                                self.sizes[i] / 2 * self.axes[i],
                                                self.center +
                                                j * self.sizes[i - 2] * self.axes[i - 2] +
                                                k * self.sizes[i - 1] * self.axes[i - 1] +
                                                self.sizes[i] / 2 * self.axes[i]]).T)
        
        return np.array(edges)
                    
    def render(self, ax):
        # drawing the center of the body
        ax.scatter(self.center[0], self.center[1], self.center[2], color="black", s=100)
        
        colors = ['red', 'blue', 'green']
        static_axes = np.eye(3)
        
        for i in range(3):
            # coordinate system associated with the body
            vector = np.array([self.center, self.center + self.axes[i]]).T
            ax.plot(vector[0], vector[1], vector[2], color=colors[i])
            
            # fixed coordinate system
            vector = np.array([self.center, self.center + static_axes[i]]).T
            ax.plot(vector[0], vector[1], vector[2], color=colors[i])
        
        del colors
        del static_axes
        del vector
        
        # drawing the body
        for edge in self.box_edges():
            ax.plot(edge[0], edge[1], edge[2], color='grey')


if __name__ == "__main__":
    box_center = np.array([0, 1, -5])
    box_sizes = np.array([3, 3, 3])
    
    box_x_axis = np.array([1, -1, 0])
    box_y_axis = np.array([-1, -1, 0.5])
    
    box_x_axis = box_x_axis / np.linalg.norm(box_x_axis)   # normalization of vector along Ox
    box_y_axis = box_y_axis / np.linalg.norm(box_y_axis)   # normalization of vector along Oy
    
    cube = Box(box_center, box_sizes, box_x_axis, box_y_axis)
    
    ax = plt.figure(figsize=(7, 7)).add_subplot(projection='3d')
    
    cube.render(ax)
