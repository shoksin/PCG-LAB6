import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

#буква "Н"
vertices = np.array([
    # Левая стойка
    [0, 0, 0], [0, 3, 0], [0, 3, 1], [0, 0, 1],
    [1, 0, 0], [1, 3, 0], [1, 3, 1], [1, 0, 1],
    # Правая стойка
    [3, 0, 0], [3, 3, 0], [3, 3, 1], [3, 0, 1],
    [4, 0, 0], [4, 3, 0], [4, 3, 1], [4, 0, 1],
    # Перекладина
    [1, 1.25, 0], [1, 1.75, 0], [1, 1.75, 1], [1, 1.25, 1],
    [3, 1.25, 0], [3, 1.75, 0], [3, 1.75, 1], [3, 1.25, 1],
])

edges = [
    # Левая стойка
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
    # Правая стойка
    (8, 9), (9, 10), (10, 11), (11, 8),
    (12, 13), (13, 14), (14, 15), (15, 12),
    (8, 12), (9, 13), (10, 14), (11, 15),
    # Перекладина
    (16, 17), (17, 18), (18, 19), (19, 16),
    (20, 21), (21, 22), (22, 23), (23, 20),
    (16, 20), (17, 21), (18, 22), (19, 23)
]

def plot_3d_wireframe(ax, vertices, edges, title=""):
    """Отрисовка 3D объекта"""
    ax.clear()
    lines = [(vertices[start], vertices[end]) for start, end in edges]
    lc = Line3DCollection(lines, colors='k', linewidths=2)
    ax.add_collection3d(lc)
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='r', s=20)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-5, 7])
    ax.set_ylim([-5, 7])
    ax.set_zlim([-5, 7])

    transformation_text = "Transformation Matrix:\n" + "\n".join([" ".join(f"{val:.2f}" for val in row) for row in transformation_matrix])
    ax.text2D(1.4, 1, transformation_text, transform=ax.transAxes, fontsize=20, verticalalignment='center')

def plot_projection(vertices, plane, title):
    """Отрисовка проекции"""
    fig, ax = plt.subplots()
    if plane == 'Oxy':
        ax.scatter(vertices[:, 0], vertices[:, 1], c='r')
        for start, end in edges:
            x = [vertices[start, 0], vertices[end, 0]]
            y = [vertices[start, 1], vertices[end, 1]]
            ax.plot(x, y, 'k')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    elif plane == 'Oxz':
        ax.scatter(vertices[:, 0], vertices[:, 2], c='r')
        for start, end in edges:
            x = [vertices[start, 0], vertices[end, 0]]
            z = [vertices[start, 2], vertices[end, 2]]
            ax.plot(x, z, 'k')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
    elif plane == 'Oyz':
        ax.scatter(vertices[:, 1], vertices[:, 2], c='r')
        for start, end in edges:
            y = [vertices[start, 1], vertices[end, 1]]
            z = [vertices[start, 2], vertices[end, 2]]
            ax.plot(y, z, 'k')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')

    ax.set_title(title)
    ax.grid(True)
    plt.show()

transformed_vertices = vertices.copy()
transformation_matrix = np.eye(4)

def apply_transformation(vertices, matrix):
    """Применить трансформацию к вершинам"""
    homogenous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))  # Однородные координаты
    transformed = homogenous @ matrix.T
    return transformed[:, :3]

def update_matrix(new_matrix):
    """Обновить глобальную матрицу трансформации"""
    global transformation_matrix
    transformation_matrix = new_matrix @ transformation_matrix

def translate(vertices, dx, dy, dz):
    """Перенос"""
    matrix = np.array([[1, 0, 0, dx],
                       [0, 1, 0, dy],
                       [0, 0, 1, dz],
                       [0, 0, 0, 1]])
    update_matrix(matrix)
    return apply_transformation(vertices, matrix)

def rotate(vertices, angle, axis):
    """Вращение вокруг оси"""
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        matrix = np.array([[1, 0, 0, 0],
                           [0, c, -s, 0],
                           [0, s, c, 0],
                           [0, 0, 0, 1]])
    elif axis == 'y':
        matrix = np.array([[c, 0, s, 0],
                           [0, 1, 0, 0],
                           [-s, 0, c, 0],
                           [0, 0, 0, 1]])
    elif axis == 'z':
        matrix = np.array([[c, -s, 0, 0],
                           [s, c, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
    update_matrix(matrix)
    return apply_transformation(vertices, matrix)

def scale(vertices, sx, sy, sz):
    """Масштабирование."""
    matrix = np.array([[sx, 0, 0, 0],
                       [0, sy, 0, 0],
                       [0, 0, sz, 0],
                       [0, 0, 0, 1]])
    update_matrix(matrix)
    return apply_transformation(vertices, matrix)


def on_key(event):
    global transformed_vertices
    if event.key == 'up':
        transformed_vertices = translate(transformed_vertices, 0, 1, 0)
    elif event.key == 'down':
        transformed_vertices = translate(transformed_vertices, 0, -1, 0)
    elif event.key == 'left': 
        transformed_vertices = translate(transformed_vertices, -1, 0, 0)
    elif event.key == 'right':
        transformed_vertices = translate(transformed_vertices, 1, 0, 0)
    elif event.key == '.':
        transformed_vertices = translate(transformed_vertices, 0, 0 ,1)
    elif event.key == ',':
        transformed_vertices = translate(transformed_vertices, 0, 0 ,-1)
    elif event.key == 'r':  # Вращение по часовой стрелке вокруг Z
        transformed_vertices = rotate(transformed_vertices, -np.pi / 18, 'z')
    elif event.key == 't':  # Вращение против часовой стрелки вокруг Z
        transformed_vertices = rotate(transformed_vertices, np.pi / 18, 'z')
    elif event.key == 'z':  # Вращение против часовой стрелки вокруг X
        transformed_vertices = rotate(transformed_vertices, np.pi / 18, 'x')
    elif event.key == 'x':  # Вращение по часовой стрелке вокруг X
        transformed_vertices = rotate(transformed_vertices, -np.pi / 18, 'x')
    elif event.key == 'c':  # Вращение против часовой стрелки вокруг Y
        transformed_vertices = rotate(transformed_vertices, np.pi / 18, 'y')
    elif event.key == 'v':  # Вращение по часовой стрелке вокруг Y
        transformed_vertices = rotate(transformed_vertices, -np.pi / 18, 'y')
    elif event.key == '=':
        transformed_vertices = scale(transformed_vertices, 1.1, 1.1, 1.1)
    elif event.key == '-':
        transformed_vertices = scale(transformed_vertices, 0.9, 0.9, 0.9)
    elif event.key == '1':
        plot_projection(transformed_vertices, 'Oxy', 'Projection on Oxy')
    elif event.key == '2':
        plot_projection(transformed_vertices, 'Oxz', 'Projection on Oxz')
    elif event.key == '3':
        plot_projection(transformed_vertices, 'Oyz', 'Projection on Oyz')

    plot_3d_wireframe(ax, transformed_vertices, edges)
    transformation_text = "Transformation Matrix:\n" + "\n".join([" ".join(f"{val:.2f}" for val in row) for row in transformation_matrix])
    ax.text2D(1.4, 1, transformation_text, transform=ax.transAxes, fontsize=20, verticalalignment='center')
    plt.draw()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(121, projection='3d')

instructions_ax = fig.add_axes([0.55, 0.1, 0.38, 0.3])
instructions_ax.add_patch(plt.Rectangle((0, 0), 1, 1, color='lightblue', lw=2, edgecolor='black'))
instructions_ax.axis('off')
instructions_ax.text(0.5, 0.5, """
Перемещение по x: ←/→
Перемещение по y: ↑/↓
Перемещение по z: ,/.
Вращение:
Вокруг Z: r,t
Вокруг X: z,x
Вокруг Y: c,v
+, -: увеличение/уменьшение объекта
1, 2, 3: проекции
""", ha='center', va='center', fontsize=16)

plot_3d_wireframe(ax, transformed_vertices, edges)

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()