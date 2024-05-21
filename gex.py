import numpy as np
from itertools import product

N = 5

"""
TODO:
- Сгенерировать матрицу с shape (N**3, 4)
- Сгенерить декартово произведение координат длины 3
- Заполнить 4-ю колонку дефолтными значениями (-1, например)

Результат - data
"""

data = np.full((N**3, 4), -1)

cartesian_product = np.array(list(product(np.arange(N), repeat=3)))

data[:, :-1] = cartesian_product

print(data)

"""
coords - координаты в формате [x, y, z]
info - растояние гексагона от центра

TODO:
- Посчитать индекс координат в массиве
- По посчитанном индексу записать info
"""
def set_data(coords: list, info: int):
  x, y, z = coords
  index = x*(N**2) + y*N + z
  data[index][-1] = info

set_data([2, 2, 2], 0)

set_data([2, 3, 1], 1)
set_data([2, 1, 3], 1)
set_data([1, 2, 3], 1)
set_data([3, 2, 1], 1)
set_data([1, 3, 2], 1)
set_data([3, 1, 2], 1)


"""
Делаем аналогичную вещь, но смещаем каждую координату
"""

def set_rel_data(coords: list, info: int):
  coordinate_offset = N//2
  x, y, z = [c+coordinate_offset for c in coords]
  index = x*(N**2) + y*N + z
  data[index][-1] = info


set_rel_data([0, 2, -2], 2)
set_rel_data([1, 1, -2], 2)
set_rel_data([2, 0, -2], 2)
set_rel_data([2, -1, -1], 2)
set_rel_data([2, -2, 0], 2)
set_rel_data([1, -2, 1], 2)
set_rel_data([0, -2, 2], 2)
set_rel_data([-1, -1, 2], 2)
set_rel_data([-2, 0, 2], 2)
set_rel_data([-2, 1, 1], 2)
set_rel_data([-2, 2, 0], 2)
set_rel_data([-1, 2, -1], 2)

import plotly.express as px
fig = px.scatter_3d(
    data,
    x=data[:, 0],
    y=data[:, 1],
    z=data[:, 2],
    color=data[:, 3]
)
fig.show()

data_to_write = data[data[:, 3] != -1]
fig = px.scatter_3d(
    data_to_write,
    x=data_to_write[:, 0],
    y=data_to_write[:, 1],
    z=data_to_write[:, 2],
    color=data_to_write[:, 3]
)
fig.show()


"""
max_dist - максимальное расстояние гексагона

TODO:
- Вычислить N в зависимости от max_dist
- Собрать код генерации куба выше в одну функцию

Примечание: расстояние от центра не добавлять, займемся этим ниже
"""

def create_map(max_dist: int):
  N = (max_dist*2) + 1
  data = np.full((N**3, 4), 1)
  cartesian_product = np.array(list(product(np.arange(N), repeat=3)))
  data[:, :-1] = cartesian_product

  return data

MAX_DIST = 20

map_data = create_map(MAX_DIST)

print(map_data)

"""
coords_map - куб с гексагонами

Функция проходит по точкам куба и проверяет соответствие условию гексагона
Возвращает массив индексов гексагонов

Для реализации лучше использовать np.where
"""

def get_hex_idxs(coords_map):
  return np.array(np.where(coords_map[:, 0] + coords_map[:, 1] + coords_map[:, 2] == 3*MAX_DIST)[0])

hexagon_idxs = get_hex_idxs(map_data)

print(hexagon_idxs)

hexagon_data = map_data[hexagon_idxs]

for hexagon in hexagon_data:
  coords = hexagon[:3]
  coords = [c-MAX_DIST for c in coords]
  hexagon[-1] = max([abs(i) for i in coords])

fig = px.scatter_3d(
    hexagon_data,
    x=hexagon_data[:, 0],
    y=hexagon_data[:, 1],
    z=hexagon_data[:, 2],
    color=hexagon_data[:, 3]
)
fig.show()

new_map = create_map(MAX_DIST)
hex_idxs = get_hex_idxs(new_map)
hex_data = new_map[hex_idxs]

fig = px.scatter_3d(
    hex_data,
    x=hex_data[:, 0],
    y=hex_data[:, 1],
    z=hex_data[:, 2],
    color=hex_data[:, 3]
)
fig.show()

ALL_MOVEMENTS = np.array([
    [0, 1, -1],
    [1, 0, -1],
    [1, -1, 0],
    [0, -1, 1],
    [-1, 0, 1],
    [-1, 1, 0]
])

import random

river_movements = [ALL_MOVEMENTS[0], ALL_MOVEMENTS[1], ALL_MOVEMENTS[3], ALL_MOVEMENTS[4]]

RIVERS_COUNT = 40

MAX_RIVER_LENGTH = 7

RIVER_TYPE = 10

np.random.seed(42)
random.seed(42)

river_map = np.copy(new_map)
size = MAX_DIST*2 + 1

for _ in range(RIVERS_COUNT):
  river_length = np.random.randint(0, MAX_RIVER_LENGTH)
  curr_idx = random.choice(hex_idxs)
  curr_data = river_map[curr_idx]
  curr_len = 0

  while curr_len < river_length:
    river_map[curr_idx][-1] = RIVER_TYPE
    r_move_idx = random.choice(np.arange(len(river_movements)))
    r_move = river_movements[r_move_idx]
    x,y,z = curr_data[:3]
    coords = np.array([x,y,z])
    new_coords = coords + r_move
    new_idx = new_coords[0]*(size**2) + new_coords[1]*size + new_coords[2]
    if new_idx not in hex_idxs:
      continue
    new_data = river_map[new_idx]
    if new_data[-1] == RIVER_TYPE:
      continue
    else:
      curr_len += 1
      curr_idx = new_idx
      curr_data = river_map[curr_idx]


hex_data = river_map[hex_idxs]

fig = px.scatter_3d(
    hex_data,
    x=hex_data[:, 0],
    y=hex_data[:, 1],
    z=hex_data[:, 2],
    color=hex_data[:, 3]
)
fig.show()

hills_map = np.copy(river_map)

HILLS_COUNT = 10

HILLS_TYPE = 100

for _ in range(HILLS_COUNT):
  curr_idx = np.random.choice(hex_idxs)
  curr_data = river_map[curr_idx]

  hills_map[curr_idx][-1] = HILLS_TYPE

  for move in ALL_MOVEMENTS:
    x,y,z = curr_data[:3]
    coords = np.array([x,y,z])
    new_coords = coords + move
    new_idx = new_coords[0]*(size**2) + new_coords[1]*size + new_coords[2]
    if new_idx not in hex_idxs:
      continue
    hills_map[new_idx][-1] = HILLS_TYPE


PLAIN_TYPE = 1

types_map = {
    PLAIN_TYPE: "Plain",
    HILLS_TYPE: "Hill",
    RIVER_TYPE: "River",
}

vfunc = np.vectorize(types_map.get)

hex_data = hills_map[hex_idxs]

colors = vfunc(hex_data[:, 3])

fig = px.scatter_3d(
    hex_data,
    x=hex_data[:, 0],
    y=hex_data[:, 1],
    z=hex_data[:, 2],
    hover_name=hex_idxs,
    color=colors,
    color_discrete_sequence=["#387C44",  "gray", "#3EA99F"],
)
fig.show()

ROUTE_TYPE = 1000

def visualize_solve(route_path):
  route_map = np.copy(hills_map)

  for cell_idx in route_path:
    route_map[cell_idx][-1] = ROUTE_TYPE

  hex_data = route_map[hex_idxs]
  vfunc = np.vectorize(types_map.get)

  colors = vfunc(hex_data[:, 3])

  fig = px.scatter_3d(
      hex_data,
      x=hex_data[:, 0],
      y=hex_data[:, 1],
      z=hex_data[:, 2],
      hover_name=hex_idxs,
      color=colors,
      color_discrete_sequence=["#387C44", "gray", "#3EA99F", "maroon"],
  )
  fig.show()

visualize_solve([59_460, 57_780, 57_740, 56_060, 54_380])

back_from_idx = 52_940
back_to_idx = 45_860

import math

def count_dist(a_idx, b_idx):
  a_data = hills_map[a_idx][:3]
  b_data = hills_map[b_idx][:3]

  return math.sqrt(sum((a_data - b_data)**2))

print(count_dist(back_from_idx, back_to_idx))

init_dist = count_dist(back_from_idx, back_to_idx)

"""
curr_idx - текущий индекс клетки
path - массив, путь до текущей клетки
prev_dist - расстояние от предыдущей клетки до итоговой
prev_movement_idx - индекс прошлого движения, используем чтобы не ходить "туда-сюда"
"""
def backtracking(curr_idx, path, prev_dist, prev_movement_idx, idx_to):
  if curr_idx == idx_to:
      return path
  curr_data = river_map[curr_idx]
  x,y,z = curr_data[:3]
  coords = np.array([x,y,z])
  i = 0
  for move in ALL_MOVEMENTS:
    if list(move) != list(ALL_MOVEMENTS[prev_movement_idx]):
      new_coords = coords + move
      new_idx = new_coords[0]*(size**2) + new_coords[1]*size + new_coords[2]
      curr_dist = count_dist(new_idx, idx_to)
      if new_idx in hex_idxs and curr_dist <= (init_dist*2) and prev_dist > curr_dist:
        return backtracking(new_idx, path+[new_idx], curr_dist, i, idx_to)
    i += 1
  return None

back_path = backtracking(back_from_idx, [back_from_idx], init_dist, -1, back_to_idx)

print(back_path)

visualize_solve(back_path)

def visualise_backtrack_solve(idx_from, idx_to):
  init_dist = count_dist(idx_from, idx_to)

  back_path = backtracking(idx_from, [idx_from], init_dist, -1, idx_to)


  if back_path is not None:
    visualize_solve(back_path)

visualise_backtrack_solve(29_380, 46_180)


greedy_path = [back_from_idx]
curr_idx = back_from_idx
init_dist = count_dist(back_from_idx, back_to_idx)

curr_dist = init_dist
curr_data = hills_map[curr_idx]
coords = np.array(curr_data[:3])

while curr_dist > 0:
  pos_moves = []
  for move in ALL_MOVEMENTS:
    new_coords = coords + move
    new_idx = new_coords[0]*(size**2) + new_coords[1]*size + new_coords[2]
    new_dist = count_dist(new_idx, back_to_idx)
    if new_dist < curr_dist:
      pos_moves.append([new_dist, new_idx, new_coords])

  curr_dist, curr_idx, coords = sorted(pos_moves)[0]
  greedy_path.append(curr_idx)

print(greedy_path)

visualize_solve(greedy_path)

def count_greedy_path(idx_from, idx_to):
  greedy_path = [idx_from]
  curr_idx = idx_from
  init_dist = count_dist(idx_from, idx_to)

  curr_dist = init_dist
  curr_data = hills_map[curr_idx]
  coords = np.array(curr_data[:3])

  while curr_dist > 0:
    pos_moves = []
    for move in ALL_MOVEMENTS:
      new_coords = coords + move
      new_idx = new_coords[0]*(size**2) + new_coords[1]*size + new_coords[2]
      new_dist = count_dist(new_idx, idx_to)
      if new_dist < curr_dist:
        pos_moves.append([new_dist, new_idx, new_coords])

    curr_dist, curr_idx, coords = sorted(pos_moves)[0]
    greedy_path.append(curr_idx)

  return greedy_path

def visualise_greedy_solve(idx_from, idx_to):
  greedy_path = count_greedy_path(idx_from, idx_to)

  visualize_solve(greedy_path)

visualise_greedy_solve(29_540, 46_220)

def count_greedy_path(idx_from, idx_to):
  greedy_path = [idx_from]
  curr_idx = idx_from
  init_dist = count_dist(idx_from, idx_to)

  curr_dist = init_dist
  curr_data = hills_map[curr_idx]
  coords = np.array(curr_data[:3])

  while curr_dist > 0:
    pos_moves = []
    for move in ALL_MOVEMENTS:
      new_coords = coords + move
      new_idx = new_coords[0]*(size**2) + new_coords[1]*size + new_coords[2]
      new_dist = count_dist(new_idx, idx_to)
      new_type = hills_map[new_idx][-1]
      if new_idx not in greedy_path and new_type != RIVER_TYPE and new_type != HILLS_TYPE:
        pos_moves.append([new_dist, new_idx, new_coords])

    curr_dist, curr_idx, coords = sorted(pos_moves)[0]
    greedy_path.append(curr_idx)

  return greedy_path

visualise_greedy_solve(44_820, 41_380)
visualise_greedy_solve(29_380, 58_060)

part_idxs = hex_idxs[hex_idxs > 51_000]

part_hex_data = hills_map[part_idxs]

colors = vfunc(part_hex_data[:, 3])

fig = px.scatter_3d(
    hex_data,
    x=part_hex_data[:, 0],
    y=part_hex_data[:, 1],
    z=part_hex_data[:, 2],
    hover_name=part_idxs,
    color=colors,
    color_discrete_sequence=["gray", "#387C44","#3EA99F"],
)
fig.show()

from_idx = 64_620
to_idx = 52_140

visualize_solve([from_idx, to_idx])

possible_movements = np.array([
    [1, 0, -1],
    [0, 1, -1],
])

part_n = 8
part_m = 19

weights = np.full((part_n, part_m), PLAIN_TYPE)

print(weights)

to_data = hills_map[to_idx]
x, y, z = to_data[:3]
to_coords = np.array([x, y, z])

for i in range(part_n):
  x_coords = to_coords + (possible_movements[0] * i)
  for j in range(part_m):
    cell_coords = x_coords + (possible_movements[1] * j)
    cell_idx = cell_coords[0]*(size**2) + cell_coords[1]*size + cell_coords[2]
    cell = hills_map[cell_idx]
    cell_type = cell[-1]

    elems_to_compare = []

    if i > 0:
      elems_to_compare.append(weights[i - 1][j])

    if j > 0:
      elems_to_compare.append(weights[i][j - 1])

    weights[i][j] = cell_type

    if len(elems_to_compare):
      weights[i][j] += min(elems_to_compare)

print(weights)

path = []
curr_i, curr_j = np.array(weights.shape) - 1

while curr_i > 0 or curr_j > 0:
  cell_coords = to_coords + (possible_movements[0] * curr_i) + (possible_movements[1] * curr_j)
  cell_idx = cell_coords[0]*(size**2) + cell_coords[1]*size + cell_coords[2]

  path.append(cell_idx)

  left_value = weights[curr_i][curr_j - 1] if curr_j > 0 else 1000
  top_value = weights[curr_i - 1][curr_j] if curr_i > 0 else 1000

  if left_value <= top_value:
    curr_j -= 1
  else:
    curr_i -= 1

path.append(to_idx)
path = np.array(path)
print(path)

visualize_solve(path)

print(hills_map[from_idx] - hills_map[to_idx])


new_from_idx = 42_140
new_to_idx = 64_300


def get_diff(idx_from, idx_to):
  return hills_map[idx_from][:3] - hills_map[idx_to][:3]

print(get_diff(new_from_idx, new_to_idx))


movement_idxs = np.where(ALL_MOVEMENTS[:, 2] == 1)[0]

movements = ALL_MOVEMENTS[movement_idxs]

print(movements, movements[:, 0:2], sep='\n')

"""
Функция возвращает движения для заданных индексов, а также индекс фиксированной переменной
"""


def get_movements(idx_from, idx_to):
    diff = get_diff(idx_from, idx_to)
    pos, neg = [], []
    for i in range(len(diff)):
        if diff[i] >= 0:
            pos.append(i)
        else:
            neg.append(i)
    if len(pos) > len(neg):
        sign = -1
        idx = neg[0]
    else:
        sign = 1
        idx = pos[0]

    movement_idxs = np.where(ALL_MOVEMENTS[:, idx] == sign)[0]
    movements = ALL_MOVEMENTS[movement_idxs]

    return movements, idx

print(get_movements(new_from_idx, new_to_idx))

"""
movements - массив движений, получаем из функции выше
fixed_var_idx - индекс зафиксированной переменной, получаем из функции выше
diff - разница координат, получаем из get_diff
"""


def get_size(movements, fixed_var_idx, diff):
    res = [0, 0]
    idxs = [0, 1, 2]
    idxs.pop(fixed_var_idx)
    for i in range(2):
        idx = idxs[i]
        value = diff[idx]
        if movements[0][idx] != 0:
            res[0] = (value // movements[0][idx]) + 1
        else:
            res[1] = (value // movements[1][idx]) + 1

    return res

diff = get_diff(new_from_idx, new_to_idx)

movements, fixed_var_idx = get_movements(new_from_idx, new_to_idx)

print(get_size(movements, fixed_var_idx, diff))


def solve_turtle_task(idx_from, idx_to):
    diff = get_diff(idx_from, idx_to)

    movements, fixed_var_idx = get_movements(idx_from, idx_to)

    size_x_y = get_size(movements, fixed_var_idx, diff)

    part_n, part_m = size_x_y

    weights = np.full((part_n, part_m), PLAIN_TYPE)

    to_data = hills_map[idx_to]
    x, y, z = to_data[:3]
    to_coords = np.array([x, y, z])

    for i in range(part_n):
        x_coords = to_coords + (movements[0] * i)
        for j in range(part_m):
            cell_coords = x_coords + (movements[1] * j)
            cell_idx = cell_coords[0] * (size ** 2) + cell_coords[1] * size + cell_coords[2]
            cell = hills_map[cell_idx]
            cell_type = cell[-1]

            elems_to_compare = []

            if i > 0:
                elems_to_compare.append(weights[i - 1][j])

            if j > 0:
                elems_to_compare.append(weights[i][j - 1])

            weights[i][j] = cell_type

            if len(elems_to_compare):
                weights[i][j] += min(elems_to_compare)

    path = []
    curr_i, curr_j = np.array(weights.shape) - 1

    while curr_i > 0 or curr_j > 0:
        cell_coords = to_coords + (movements[0] * curr_i) + (movements[1] * curr_j)
        cell_idx = cell_coords[0] * (size ** 2) + cell_coords[1] * size + cell_coords[2]

        path.append(cell_idx)

        left_value = weights[curr_i][curr_j - 1] if curr_j > 0 else 1000
        top_value = weights[curr_i - 1][curr_j] if curr_i > 0 else 1000

        if left_value <= top_value:
            curr_j -= 1
        else:
            curr_i -= 1

    path.append(idx_to)
    path = np.array(path)

    return path

new_path = solve_turtle_task(new_from_idx, new_to_idx)

print(new_path)

visualize_solve(new_path)

def draw_map_with_turtle_route(from_idx, to_idx):
  solve_path = solve_turtle_task(from_idx, to_idx)
  visualize_solve(solve_path)

draw_map_with_turtle_route(57_780, 52_420)

draw_map_with_turtle_route(37_620, 51_500)
