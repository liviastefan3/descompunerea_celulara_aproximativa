import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Dimensiunea hărții și a unei celule
LATIME = 400
INALTIME = 300
CELULA = 10   # => grila va avea 40 x 30 celule

# Obstacole: (x_min, y_min, x_max, y_max)
obstacole = [
    (100, 200, 300, 240),
    (160, 80, 200, 200),
    (250, 40, 280, 160),
]

# Start și goal, în coordonate (x, y)
start_xy = (20, 100)
goal_xy = (370, 20)
#construirea grilei

n_rows = INALTIME // CELULA
n_cols = LATIME // CELULA


def celula_atinge_obstacol(row, col):
    """True dacă celula se suprapune (chiar și parțial) cu un obstacol."""
    x1 = col * CELULA
    x2 = x1 + CELULA
    y1 = row * CELULA
    y2 = y1 + CELULA
    for (ox1, oy1, ox2, oy2) in obstacole:
        if x1 < ox2 and x2 > ox1 and y1 < oy2 and y2 > oy1:
            return True
    return False


# 0 = liber, 1 = ocupat
grid = np.zeros((n_rows, n_cols), dtype=int)
for row in range(n_rows):
    for col in range(n_cols):
        if celula_atinge_obstacol(row, col):
            grid[row][col] = 1
#start si goaL ca celule

def coord_la_celula(punct):
    x, y = punct
    return (int(y // CELULA), int(x // CELULA))   # (row, col)


start = coord_la_celula(start_xy)
goal = coord_la_celula(goal_xy)

if grid[start] == 1 or grid[goal] == 1:
    raise ValueError("Start sau goal este într-un obstacol!")
#cautarea drumului
def bfs(grid, start, goal):
    coada = deque([start])
    parinte = {start: None}    # de unde am ajuns în fiecare celulă

    while coada:
        curent = coada.popleft()
        if curent == goal:
            break
        r, c = curent
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < n_rows and 0 <= nc < n_cols
                    and grid[nr][nc] == 0
                    and (nr, nc) not in parinte):
                parinte[(nr, nc)] = curent
                coada.append((nr, nc))

    if goal not in parinte:
        return None            # nu există drum

    # reconstruim drumul mergând înapoi de la goal la start
    drum = []
    nod = goal
    while nod is not None:
        drum.append(nod)
        nod = parinte[nod]
    drum.reverse()
    return drum
#functia de afisare
def deseneaza(drum=None):
    fig, ax = plt.subplots(figsize=(10, 7.5))

    # celulele ocupate (negre) și libere (albe)
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1, origin="lower",
              extent=(0, LATIME, 0, INALTIME))

    # liniile grilei
    ax.set_xticks(np.arange(0, LATIME + 1, CELULA))
    ax.set_yticks(np.arange(0, INALTIME + 1, CELULA))
    ax.grid(True, color="gray", linewidth=0.5)
    ax.tick_params(labelbottom=False, labelleft=False)

    # drumul (prin centrele celulelor)
    if drum:
        xs = [(c + 0.5) * CELULA for (r, c) in drum]
        ys = [(r + 0.5) * CELULA for (r, c) in drum]
        ax.plot(xs, ys, color="orange", linewidth=3)

    # start și goal
    ax.plot((start[1] + 0.5) * CELULA, (start[0] + 0.5) * CELULA,
            "bo", markersize=12)
    ax.plot((goal[1] + 0.5) * CELULA, (goal[0] + 0.5) * CELULA,
            "ro", markersize=12)

    ax.set_title("Descompunere omogenă")
    plt.show()

#cautarea drumului
def bfs(grid, start, goal):
    coada = deque([start])
    parinte = {start: None}    # de unde am ajuns în fiecare celulă

    while coada:
        curent = coada.popleft()
        if curent == goal:
            break
        r, c = curent
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < n_rows and 0 <= nc < n_cols
                    and grid[nr][nc] == 0
                    and (nr, nc) not in parinte):
                parinte[(nr, nc)] = curent
                coada.append((nr, nc))

    if goal not in parinte:
        return None            # nu există drum

    # reconstruim drumul mergând înapoi de la goal la start
    drum = []
    nod = goal
    while nod is not None:
        drum.append(nod)
        nod = parinte[nod]
    drum.reverse()
    return drum
drum = bfs(grid, start, goal)

if drum is None:
    print("Nu există drum în această rezoluție.")
else:
    print(f"Drum găsit, cu {len(drum)} celule.")

deseneaza(drum)