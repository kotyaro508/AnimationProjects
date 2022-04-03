import matplotlib.pyplot as plt
import numpy as np


# угол между векторами u и v при правом повороте вокруг оси z
def angle_between(u, v, z):
    u_norm = u / np.linalg.norm(u)
    v_norm = v / np.linalg.norm(v)
    cross = np.cross(u_norm, v_norm)
    if np.linalg.norm(cross) == 0:
        return np.arccos(np.dot(u_norm, v_norm)) % (2 * np.pi)
    return (np.arccos(np.dot(u_norm, v_norm)) *
            np.dot(z / np.linalg.norm(z), cross / np.linalg.norm(cross))) % (2 * np.pi)


class Box:
    def __init__(self, center, sizes, x_axis, y_axis):
        self.center = center                                               # центр тела
        self.axes = np.array([x_axis, y_axis, np.cross(x_axis, y_axis)])   # координатные оси
        self.sizes = sizes                                                 # длины сторон
        
        # вектор линии узлов == пересечение двух плоскостей Oxy == векторное произведение двух осей z
        knot_line_axis = (np.cross(np.array([0, 0, 1]), self.axes[2]) /
                          np.linalg.norm(np.cross(np.array([0, 0, 1]),
                                                  self.axes[2])))
        
        # один из двух углов прецессии (в множестве [0; pi) или в множестве (pi; 2pi) )
        precession = angle_between(np.array([1, 0, 0]),
                                   knot_line_axis,
                                   np.array([0, 0, 1]))
        
        self.precession = precession % np.pi                               # угол прецессии (в множестве [0; pi) )
        
        # задание вектора линии узлов под острым углом прецессии к неподвижной оси x (умножение на 1 или -1)
        knot_line_axis *= (-2) * (precession // np.pi - 0.5)
        
        del precession
        
        self.nutation = angle_between(np.array([0, 0, 1]),                 # угол нутации (в множестве [0; 2pi) )
                                      self.axes[2],
                                      knot_line_axis)
        
        self.own_rotation = angle_between(knot_line_axis,                  # угол собственного вращения
                                          self.axes[0],                     # (в множестве [0; 2pi) )
                                          self.axes[2])
        
        del knot_line_axis
    
    def box_edges(self):
        edges = []                                                         # рёбра тела
        
        # добавляем по 4 ребра вдоль каждой оси
        for i in range(3):
            for j in (-0.5, 0.5):
                for k in (-0.5, 0.5):
                    
                # индексы j и k задают расположение центров каждого из 4-х рёбер вдоль оси i
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
        # отрисовка центра тела
        ax.scatter(self.center[0], self.center[1], self.center[2], color="black", s=100)
        
        colors = ['red', 'blue', 'green']
        static_axes = np.eye(3)
        
        for i in range(3):
            # система координат, связанная с телом
            vector = np.array([self.center, self.center + self.axes[i]]).T
            ax.plot(vector[0], vector[1], vector[2], color=colors[i])
            
            # неподвижная система координат
            vector = np.array([self.center, self.center + static_axes[i]]).T
            ax.plot(vector[0], vector[1], vector[2], color=colors[i])
        
        del colors
        del static_axes
        del vector
        
        # отрисовка тела
        for edge in self.box_edges():
            ax.plot(edge[0], edge[1], edge[2], color='grey')


if __name__ == "__main__":
    box_center = np.array([0, 1, -5])
    box_sizes = np.array([3, 3, 3])
    
    box_x_axis = np.array([1, -1, 0])
    box_y_axis = np.array([-1, -1, 0.5])
    
    box_x_axis = box_x_axis / np.linalg.norm(box_x_axis)   # нормировка вектора вдоль Ox
    box_y_axis = box_y_axis / np.linalg.norm(box_y_axis)   # нормировка вектора вдоль Oy
    
    cube = Box(box_center, box_sizes, box_x_axis, box_y_axis)
    
    ax = plt.figure(figsize=(7, 7)).add_subplot(projection='3d')
    
    cube.render(ax)
