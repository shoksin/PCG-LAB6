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

# Грани буквы "Н"
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
    if vertices.shape[1] == 2:  # Для проекции 2D
        ax.plot(vertices[:, 0], vertices[:, 1], 'bo-', markersize=4)  # Рисуем точки
    else:  # Для 3D
        for face in faces:
            poly = vertices[face]
            ax.add_collection3d(Poly3DCollection([poly], alpha=0.5, edgecolor='k'))
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-5, 10])
    ax.set_ylim([-5, 10])
    ax.set_zlim([-5, 5])


def update_matrix_display(ax, matrix):
    """Обновить отображение матрицы на экране."""
    ax.clear()
    ax.axis('off')
    matrix_text = '\n'.join([' '.join([f"{value:.2f}" for value in row]) for row in matrix])
    
    # Используем панель для отображения матрицы
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color='lightgray', lw=2, edgecolor='black'))
    ax.text(0.5, 0.5, matrix_text, fontsize=12, va='center', ha='center', transform=ax.transAxes)
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
    """Обработчик нажатий клавиш."""
    global transformed_vertices
    if event.key == 'up':  # Вверх (стрелочка)
        transformed_vertices = translate(transformed_vertices, 0, 1, 0)
    elif event.key == 'down':  # Вниз (стрелочка)
        transformed_vertices = translate(transformed_vertices, 0, -1, 0)
    elif event.key == 'left':  # Влево (стрелочка)
        transformed_vertices = translate(transformed_vertices, -1, 0, 0)
    elif event.key == 'right':  # Вправо (стрелочка)
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
    elif event.key == 'w':  # Масштабирование (увеличение)
        transformed_vertices = scale(transformed_vertices, 1.1, 1.1, 1.1)
    elif event.key == 's':  # Масштабирование (уменьшение)
        transformed_vertices = scale(transformed_vertices, 0.9, 0.9, 0.9)

    # Обновить отображение
    plot_3d_object(ax, transformed_vertices, faces)
    update_matrix_display(matrix_ax, transformation_matrix)
    plt.draw()

def on_projection_key(event):
    """Обработчик нажатий для проекций."""
    global transformed_vertices
    if event.key == '1':
        projected = project(transformed_vertices, plane='Oxy')
        plot_3d_object(ax, projected, faces)
    elif event.key == '2':
        projected = project(transformed_vertices, plane='Oxz')
        plot_3d_object(ax, projected, faces)
    elif event.key == '3':
        projected = project(transformed_vertices, plane='Oyz')
        plot_3d_object(ax, projected, faces)

# Создание интерфейса
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')  # График 3D объекта
matrix_ax = fig.add_subplot(122)  # Отображение матрицы

# Построить начальное состояние
plot_3d_object(ax, transformed_vertices, faces)
update_matrix_display(matrix_ax, transformation_matrix)

# Добавление графического поля с текстом инструкций
instructions_ax = fig.add_axes([0.7, 0.1, 0.28, 0.3], frameon=False)
instructions_ax.add_patch(plt.Rectangle((0, 0), 1, 1, color='lightblue', lw=2, edgecolor='black'))
instructions_ax.text(0.5, 0.5, """
Стрелочки:
  Вверх - переместить вверх
  Вниз - переместить вниз
  Влево - переместить влево
  Вправо - переместить вправо

r, t, z, x, c, v: вращение

w, s: масштабирование

1, 2, 3: проекции
""", ha='center', va='center', fontsize=12)

# Привязка событий
fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('key_press_event', on_projection_key)

plt.show()
