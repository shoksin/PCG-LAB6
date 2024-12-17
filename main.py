import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Вершины буквы "Н"
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

#Грани
faces = [
    [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7],
    [8, 9, 13, 12], [9, 10, 14, 13], [10, 11, 15, 14], [11, 8, 12, 15], [8, 9, 10, 11], [12, 13, 14, 15],
    [16, 17, 21, 20], [17, 18, 22, 21], [18, 19, 23, 22], [19, 16, 20, 23], [16, 17, 18, 19], [20, 21, 22, 23]
]


# Изначально объект не трансформирован
transformed_vertices = vertices.copy()
transformation_matrix = np.eye(4)  # Единичная матрица 4x4

def plot_3d_object(ax, vertices, faces, title="3D Object"):
    """Отрисовка 3D объекта (с учетом 2D-проекции)."""
    ax.clear()
    if vertices.shape[1] == 2:  #2D
        ax.plot(vertices[:, 0], vertices[:, 1], 'bo-', markersize=4)  # Рисуем точки
    else:  #3D
        for face in faces:
            poly = vertices[face]
            ax.add_collection3d(Poly3DCollection([poly], alpha=0.5, edgecolor='k'))
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-7, 8])
    ax.set_ylim([-7, 8])
    ax.set_zlim([-7, 7])


def update_matrix_display(ax, matrix):
    """Обновить отображение матрицы на экране."""
    ax.clear()
    ax.axis('off')
    matrix_text = '\n'.join([' '.join([f"{value:.2f}" for value in row]) for row in matrix])
    
    # Используем панель для отображения матрицы
    ax.add_patch(plt.Rectangle((0.0, 0.6), 1, 1, color='lightgray', lw=2, edgecolor='black'))
    ax.text(0.5, 0.8, matrix_text, fontsize=22, va='center', ha='center', transform=ax.transAxes)
    ax.set_title("Transformation Matrix")

def apply_transformation(vertices, matrix):
    """Применить трансформацию к вершинам."""
    homogenous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))  # Однородные координаты
    transformed = homogenous @ matrix.T
    return transformed[:, :3]

def update_matrix(new_matrix):
    """Обновить глобальную матрицу трансформации."""
    global transformation_matrix
    transformation_matrix = new_matrix @ transformation_matrix

def translate(vertices, dx, dy, dz):
    """Перенос."""
    matrix = np.array([[1, 0, 0, dx],
                       [0, 1, 0, dy],
                       [0, 0, 1, dz],
                       [0, 0, 0, 1]])
    update_matrix(matrix)
    return apply_transformation(vertices, matrix)

def rotate(vertices, angle, axis):
    """Вращение вокруг оси."""
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

def project(vertices, plane='Oxy'):
    """Проекция на плоскость."""
    if plane == 'Oxy':
        # Проекция на плоскость Oxy, где координаты Z игнорируются
        return vertices[:, :2]  # Оставляем только X и Y
    elif plane == 'Oxz':
        # Проекция на плоскость Oxz, где координаты Y игнорируются
        return vertices[:, [0, 2]]  # Оставляем только X и Z
    elif plane == 'Oyz':
        # Проекция на плоскость Oyz, где координаты X игнорируются
        return vertices[:, 1:]  # Оставляем только Y и Z

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
    elif event.key == '=':  # увеличение
        transformed_vertices = scale(transformed_vertices, 1.1, 1.1, 1.1)
    elif event.key == '-':  # уменьшение
        transformed_vertices = scale(transformed_vertices, 0.9, 0.9, 0.9)

    plot_3d_object(ax, transformed_vertices, faces)
    update_matrix_display(matrix_ax, transformation_matrix)
    plt.draw()

def on_projection_key(event):
    """Обработчик нажатий для отображения проекций."""
    global transformed_vertices

    if event.key in ['1', '2', '3']:
        plane = {'1': 'Oxy', '2': 'Oxz', '3': 'Oyz'}[event.key]
        
        projected = project(transformed_vertices, plane=plane)
        
        projection_fig, projection_ax = plt.subplots(figsize=(6, 6))
        projection_ax.set_aspect('equal', adjustable='datalim')
        
        projection_ax.plot(projected[:, 0], projected[:, 1], 'bo-', markersize=4, label=f"Projection on {plane}")
        
        if plane == 'Oxy':
            projection_ax.set_xlabel('X')
            projection_ax.set_ylabel('Y')
        elif plane == 'Oxz':
            projection_ax.set_xlabel('X')
            projection_ax.set_ylabel('Z')
        elif plane == 'Oyz':
            projection_ax.set_xlabel('Y')
            projection_ax.set_ylabel('Z')

        projection_ax.set_title(f"Projection on {plane}")
        projection_ax.legend()
        projection_ax.grid(True)

        plt.show()


fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
matrix_ax = fig.add_subplot(122)

plot_3d_object(ax, transformed_vertices, faces)
update_matrix_display(matrix_ax, transformation_matrix)

instructions_ax = fig.add_axes([0.55, 0.1, 0.38, 0.3])
instructions_ax.add_patch(plt.Rectangle((0, 0), 1, 1, color='lightblue', lw=2, edgecolor='black'))
instructions_ax.axis('off')
instructions_ax.text(0.5, 0.5, """
Стрелочки:
перемещение в плоскости Oxy
r, t, z, x, c, v: вращение
+, -: увеличение/уменьшение объекта
1, 2, 3: проекции
""", ha='center', va='center', fontsize=16)

# Привязка событий
fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('key_press_event', on_projection_key)

plt.show()
