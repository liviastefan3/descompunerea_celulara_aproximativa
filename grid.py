
import numpy as np                    # numpy: lucrează cu matrice (grila noastră)
import matplotlib.pyplot as plt       # matplotlib: desenează imaginea finală
from collections import deque         # deque: o coadă rapidă, necesară la BFS
from shapely.geometry import Polygon, box
# Polygon = formă geometrică dată prin vârfuri (obstacolele)
# box     = dreptunghi (îl folosim pentru a reprezenta o celulă)

# 2. SETĂRI GENERALE

LATIME = 400      # lățimea hărții, în unități (ca niște pixeli)
INALTIME = 300    # înălțimea hărții
CELULA = 10       # latura unei celule => 400/10 = 40 coloane, 300/10 = 30 rânduri

# Lista obstacolelor. Fiecare Polygon primește lista vârfurilor (x, y),
# în ordinea în care se parcurge conturul.
obstacole = [
    # bara de sus a lui T: 4 vârfuri => dreptunghi
    Polygon([(100, 200), (300, 200), (300, 240), (100, 240)]),
    # bara verticală a lui T
    Polygon([(250, 40), (280, 40), (280, 200), (250, 200)]),
    # triunghiul din stânga: 3 vârfuri
    Polygon([(60, 70), (180, 70), (160, 120)]),
    # triunghiul îngust din dreapta
    Polygon([(300, 20), (320, 20), (310, 150)]),
]
#de pus obstacole cu drag and drop plus ne de casute pe harta cu nr de ex 30 40
#de puus texte cu ce e de afcut de ex cum se face un dreptunghi, cum se inchide
#de facut o harta comuna aleg met 1 are desc omog aleg met 2 are desc arb
# Start și goal ca puncte pe hartă, în coordonate (x, y)
start_xy = (20, 100)     # x = 20 (stânga), y = 100
goal_xy = (370, 20)      # x = 370 (aproape de dreapta), y = 20 (jos)

# 3. CONSTRUIREA GRILEI

n_rows = INALTIME // CELULA   # numărul de rânduri; // = împărțire întreagă => 30
n_cols = LATIME // CELULA     # numărul de coloane => 40

def celula_atinge_obstacol(row, col):
    """True dacă celula (row, col) se suprapune, chiar și parțial, cu un obstacol."""
    x1 = col * CELULA    # coordonata x a colțului stânga-jos al celulei
                         # (coloana 3 => x = 30)
    y1 = row * CELULA    # coordonata y a colțului stânga-jos (rândul 5 => y = 50)

    # Construim pătratul celulei: box(x_min, y_min, x_max, y_max)
    celula = box(x1, y1, x1 + CELULA, y1 + CELULA)

    for obstacol in obstacole:           # luăm obstacolele pe rând
        if celula.intersects(obstacol):  # au măcar un punct comun?
            return True                  # da => celula e ocupată, ne oprim aici
    return False                         # niciun obstacol nu o atinge => liberă


# Creăm matricea grilei: 30 rânduri x 40 coloane, plină cu 0
# Convenție: 0 = liber, 1 = ocupat
grid = np.zeros((n_rows, n_cols), dtype=int)

for row in range(n_rows):                    # pentru fiecare rând (0..29)
    for col in range(n_cols):                # pentru fiecare coloană (0..39)
        if celula_atinge_obstacol(row, col): # celula e atinsă de un obstacol?
            grid[row][col] = 1               # da => o marcăm ocupată


# 4. START ȘI GOAL CA CELULE

def coord_la_celula(punct):
    """Convertește un punct (x, y) în celula (rând, coloană) în care se află."""
    x, y = punct                           # desfacem punctul în x și y
    return (int(y // CELULA), int(x // CELULA))
    # y // CELULA => rândul; x // CELULA => coloana
    # Atenție: ordinea se inversează! Punctul e (x, y), celula e (rând, coloană)


start = coord_la_celula(start_xy)    # (20, 100) -> (10, 2)
goal = coord_la_celula(goal_xy)      # (370, 20) -> (2, 37)

# Verificare de siguranță: robotul nu poate începe/termina într-un obstacol
if grid[start] == 1 or grid[goal] == 1:
    raise ValueError("Start sau goal este într-un obstacol!")  # oprește programul


# 5. CĂUTAREA DRUMULUI (BFS)

def bfs(grid, start, goal):
    """Găsește un drum de la start la goal prin celule libere.
    Întoarce lista de celule a drumului sau None dacă nu există."""

    coada = deque([start])     # celulele descoperite, dar încă neexplorate
    parinte = {start: None}    # celulă -> celula din care am ajuns în ea
                               # (în start n-am venit de nicăieri => None)
                               # servește și ca listă a celulelor deja văzute

    while coada:                       # cât timp mai avem celule de explorat
        curent = coada.popleft()       # scoatem prima celulă din coadă
        if curent == goal:             # am ajuns la țintă?
            break                      # da => oprim explorarea

        r, c = curent                  # desfacem celula în rând și coloană

        # Cele 4 direcții: jos, sus, dreapta, stânga (schimbarea de rând, coloană)
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc    # coordonatele vecinului (nr = nou rând)

            if (0 <= nr < n_rows and 0 <= nc < n_cols  # 1) e în interiorul hărții?
                    and grid[nr][nc] == 0              # 2) e celulă liberă?
                    and (nr, nc) not in parinte):      # 3) nu a fost deja văzută?
                parinte[(nr, nc)] = curent    # notăm din ce celulă am ajuns aici
                coada.append((nr, nc))        # îl punem în coadă, de explorat

    # Dacă goal nu a fost niciodată descoperit, nu există drum
    if goal not in parinte:
        return None

    # Reconstruim drumul mergând ÎNAPOI, de la goal spre start
    drum = []                  # lista în care strângem celulele drumului
    nod = goal                 # pornim de la goal
    while nod is not None:     # până ajungem la start (părintele lui e None)
        drum.append(nod)       # adăugăm celula curentă în listă
        nod = parinte[nod]     # sărim la părintele ei
    drum.reverse()             # lista e goal->start, o întoarcem: start->goal
    return drum


# ============================================================
# 6. AFIȘAREA
# ============================================================
def deseneaza(drum=None):
    """Desenează harta, drumul (dacă există), start, goal și obstacolele."""

    fig, ax = plt.subplots(figsize=(10, 7.5))   # creăm fereastra (fig) și zona de desen (ax)

    # Afișăm matricea ca imagine: 0 = alb, 1 = negru (cmap="Greys")
    # origin="lower"  => rândul 0 apare jos, ca pe o axă y normală
    # extent          => imaginea acoperă coordonatele 0..LATIME și 0..INALTIME
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1, origin="lower",
              extent=(0, LATIME, 0, INALTIME))

    # Liniile grilei: câte o linie la fiecare CELULA
    ax.set_xticks(np.arange(0, LATIME + 1, CELULA))   # poziții pe axa x
    ax.set_yticks(np.arange(0, INALTIME + 1, CELULA)) # poziții pe axa y
    ax.grid(True, color="gray", linewidth=0.5)        # desenăm liniile
    ax.tick_params(labelbottom=False, labelleft=False)  # ascundem numerele de pe axe

    # Drumul, dacă există
    if drum:
        xs = [(c + 0.5) * CELULA for (r, c) in drum]  # x-ul centrului fiecărei celule
        ys = [(r + 0.5) * CELULA for (r, c) in drum]  # y-ul centrului (+0.5 = mijloc)
        ax.plot(xs, ys, color="orange", linewidth=3)  # linia portocalie

    # Start (cerc albastru "bo") și goal (cerc roșu "ro") în centrul celulei lor
    ax.plot((start[1] + 0.5) * CELULA, (start[0] + 0.5) * CELULA,
            "bo", markersize=12)       # start[1] = coloana (x), start[0] = rândul (y)
    ax.plot((goal[1] + 0.5) * CELULA, (goal[0] + 0.5) * CELULA,
            "ro", markersize=12)

    # Obstacolele reale, în albastru semitransparent, peste celulele negre
    for obstacol in obstacole:
        xs, ys = obstacol.exterior.xy          # coordonatele conturului poligonului
        ax.fill(xs, ys, color="blue", alpha=0.6)  # umple poligonul (alpha = transparență)

    ax.set_title("Descompunere omogenă")       # titlul ferestrei
    plt.show()                                 # afișează fereastra


# ============================================================
# 7. PROGRAMUL PRINCIPAL (aici pornește efectiv execuția)
# ============================================================
drum = bfs(grid, start, goal)      # căutăm drumul cu BFS

if drum is None:                   # BFS n-a găsit nimic
    print("Nu există drum în această rezoluție.")
else:
    print(f"Drum găsit, cu {len(drum)} celule.")   # câte celule are drumul

deseneaza(drum)                    # desenăm tot, inclusiv drumul
