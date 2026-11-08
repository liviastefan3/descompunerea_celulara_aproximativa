import heapq
import math
import matplotlib

matplotlib.use("macosx")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

# Dimensiunea hărții (aceeași ca la descompunerea omogenă)
LATIME = 400
INALTIME = 300
ADANCIME_MAX = 6   # celula minimă: 400/2^6 x 300/2^6 = 6.25 x 4.69

obstacole = [
    # bara de sus a lui T
    Polygon([(100, 200), (300, 200), (300, 240), (100, 240)]),
    # bara verticală a lui T
    Polygon([(250, 40), (280, 40), (280, 200), (250, 200)]),
    # triunghiul din stânga
    Polygon([(60, 70), (180, 70), (160, 120)]),
    # triunghiul îngust din dreapta
    Polygon([(300, 20), (320, 20), (310, 150)]),
]
uniune_obstacole = unary_union(obstacole)

# Start și goal, în coordonate (x, y)
start_xy = (20, 100)
goal_xy = (370, 20)

LIBER, OCUPAT, MIXT = "liber", "ocupat", "mixt"


# ---------------------------------------------------------------
# Construirea arborelui (quadtree)
# ---------------------------------------------------------------
class Celula:
    def __init__(self, x, y, w, h, adancime):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.adancime = adancime
        self.copii = []                 # 4 copii dacă a fost împărțită
        self.stare = self.clasifica()

    def dreptunghi(self):
        return box(self.x, self.y, self.x + self.w, self.y + self.h)

    def clasifica(self):
        d = self.dreptunghi()
        if not d.intersects(uniune_obstacole):
            return LIBER                # nu atinge niciun obstacol
        if uniune_obstacole.contains(d):
            return OCUPAT               # complet în interiorul obstacolelor
        return MIXT                     # parțial liberă, parțial ocupată

    def centru(self):
        return (self.x + self.w / 2, self.y + self.h / 2)

    def contine(self, punct):
        px, py = punct
        return (self.x <= px <= self.x + self.w and
                self.y <= py <= self.y + self.h)


def construieste_arbore(celula):
    """Împarte recursiv celulele mixte în 4, până la ADANCIME_MAX."""
    if celula.stare != MIXT or celula.adancime == ADANCIME_MAX:
        return
    w2, h2 = celula.w / 2, celula.h / 2
    d = celula.adancime + 1
    celula.copii = [
        Celula(celula.x,      celula.y,      w2, h2, d),   # stânga-jos
        Celula(celula.x + w2, celula.y,      w2, h2, d),   # dreapta-jos
        Celula(celula.x,      celula.y + h2, w2, h2, d),   # stânga-sus
        Celula(celula.x + w2, celula.y + h2, w2, h2, d),   # dreapta-sus
    ]
    for copil in celula.copii:
        construieste_arbore(copil)


def frunze(celula, adancime_limita=ADANCIME_MAX):
    """Frunzele arborelui, văzut doar până la o anumită adâncime."""
    if not celula.copii or celula.adancime == adancime_limita:
        return [celula]
    rezultat = []
    for copil in celula.copii:
        rezultat.extend(frunze(copil, adancime_limita))
    return rezultat


radacina = Celula(0, 0, LATIME, INALTIME, 0)
construieste_arbore(radacina)

# La adâncimea maximă, celulele încă mixte sunt considerate ocupate
# (abordare conservatoare: robotul nu intră unde nu e sigur).
celule_libere = [c for c in frunze(radacina) if c.stare == LIBER]


# ---------------------------------------------------------------
# Graful de vecinătate dintre celulele libere
# ---------------------------------------------------------------
EPS = 1e-9


def latura_comuna(a, b):
    """Mijlocul laturii comune dacă a și b sunt vecine, altfel None."""
    # vecine pe orizontală (una lângă alta)
    if abs(a.x + a.w - b.x) < EPS or abs(b.x + b.w - a.x) < EPS:
        jos = max(a.y, b.y)
        sus = min(a.y + a.h, b.y + b.h)
        if sus - jos > EPS:
            x = b.x if abs(a.x + a.w - b.x) < EPS else a.x
            return (x, (jos + sus) / 2)
    # vecine pe verticală (una deasupra celeilalte)
    if abs(a.y + a.h - b.y) < EPS or abs(b.y + b.h - a.y) < EPS:
        st = max(a.x, b.x)
        dr = min(a.x + a.w, b.x + b.w)
        if dr - st > EPS:
            y = b.y if abs(a.y + a.h - b.y) < EPS else a.y
            return ((st + dr) / 2, y)
    return None


vecini = {i: [] for i in range(len(celule_libere))}
for i in range(len(celule_libere)):
    for j in range(i + 1, len(celule_libere)):
        m = latura_comuna(celule_libere[i], celule_libere[j])
        if m is not None:
            vecini[i].append((j, m))
            vecini[j].append((i, m))


def gaseste_celula(punct):
    for i, c in enumerate(celule_libere):
        if c.contine(punct):
            return i
    return None


start = gaseste_celula(start_xy)
goal = gaseste_celula(goal_xy)
if start is None or goal is None:
    raise ValueError("Start sau goal este într-un obstacol!")


# ---------------------------------------------------------------
# Căutarea drumului (A*)
# Drumul trece: centrul celulei -> jumătatea muchiei comune -> centrul
# celulei vecine. Costul unui pas = lungimea acestor două segmente.
# ---------------------------------------------------------------
def distanta(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def cost_pas(i, j, mijloc):
    """centru(i) -> jumătatea muchiei comune -> centru(j)"""
    return (distanta(celule_libere[i].centru(), mijloc) +
            distanta(mijloc, celule_libere[j].centru()))


def a_star(start, goal):
    def h(i):   # euristica: distanța în linie dreaptă până la goal
        return distanta(celule_libere[i].centru(), goal_xy)

    coada = [(h(start), 0.0, start)]
    cost = {start: 0.0}
    parinte = {start: (None, None)}     # celulă -> (celula anterioară, mijloc)

    while coada:
        _, g, curent = heapq.heappop(coada)
        if curent == goal:
            break
        if g > cost[curent]:
            continue
        for vecin, mijloc in vecini[curent]:
            g_nou = g + cost_pas(curent, vecin, mijloc)
            if vecin not in cost or g_nou < cost[vecin]:
                cost[vecin] = g_nou
                parinte[vecin] = (curent, mijloc)
                heapq.heappush(coada, (g_nou + h(vecin), g_nou, vecin))

    if goal not in parinte:
        return None, None

    # reconstruim drumul mergând înapoi de la goal la start
    celule, mijloace = [], []
    nod = goal
    while nod is not None:
        celule.append(nod)
        anterior, mijloc = parinte[nod]
        if mijloc is not None:
            mijloace.append(mijloc)
        nod = anterior
    celule.reverse()
    mijloace.reverse()
    return celule, mijloace


def puncte_drum(celule, mijloace):
    """start -> C0 -> M01 -> C1 -> M12 -> ... -> Cn -> goal"""
    puncte = [start_xy, celule_libere[celule[0]].centru()]
    for mijloc, c in zip(mijloace, celule[1:]):
        puncte.append(mijloc)
        puncte.append(celule_libere[c].centru())
    puncte.append(goal_xy)
    return puncte


# ---------------------------------------------------------------
# Afișare
# ---------------------------------------------------------------
CULORI = {LIBER: "white", OCUPAT: "#9e9e9e", MIXT: "#fff3a0"}


def deseneaza_celule(ax, celule, mixt_ca_ocupat=False):
    for c in celule:
        stare = c.stare
        if mixt_ca_ocupat and stare == MIXT:
            stare = OCUPAT
        ax.add_patch(Rectangle((c.x, c.y), c.w, c.h,
                               facecolor=CULORI[stare],
                               edgecolor="red", linewidth=0.5))
    for obstacol in obstacole:
        xs, ys = obstacol.exterior.xy
        ax.fill(xs, ys, color="blue", alpha=0.6)
    ax.plot(*start_xy, "bo", markersize=8)
    ax.plot(*goal_xy, "ro", markersize=8)
    ax.set_xlim(0, LATIME)
    ax.set_ylim(0, INALTIME)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def deseneaza_pasi():
    """Împărțirea pas cu pas, ca în figurile din cerință."""
    fig, axe = plt.subplots(2, 3, figsize=(15, 8))
    for k, ax in enumerate(axe.flat, start=1):
        deseneaza_celule(ax, frunze(radacina, adancime_limita=k))
        ax.set_title(f"Pasul {k} (adâncime {k})")
    fig.suptitle("Metoda arborelui – subdivizarea celulelor mixte "
                 "(galben = mixt, gri = ocupat, alb = liber)")
    plt.tight_layout()


def deseneaza_rezultat(celule, mijloace):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    deseneaza_celule(ax, frunze(radacina), mixt_ca_ocupat=True)
    if celule:
        # celulele prin care trece drumul, evidențiate
        for i in celule:
            c = celule_libere[i]
            ax.add_patch(Rectangle((c.x, c.y), c.w, c.h,
                                   facecolor="#ffe0b2", edgecolor="red",
                                   linewidth=0.5))
        xs, ys = zip(*puncte_drum(celule, mijloace))
        ax.plot(xs, ys, color="orange", linewidth=2.5, zorder=3)
        # centrele celulelor
        cx, cy = zip(*[celule_libere[i].centru() for i in celule])
        ax.plot(cx, cy, "ks", markersize=5, zorder=4,
                label="centrul celulei")
        # jumătățile muchiilor comune
        if mijloace:
            mx, my = zip(*mijloace)
            ax.plot(mx, my, "o", color="green", markersize=6, zorder=4,
                    label="jumătatea muchiei comune")
        ax.legend(loc="upper right")
    ax.set_title("Metoda arborelui (quadtree)")


drum_celule, drum_mijloace = a_star(start, goal)

toate = frunze(radacina)
print(f"Frunze în arbore: {len(toate)} "
      f"(față de {(LATIME // 10) * (INALTIME // 10)} celule în grila omogenă)")
print(f"Celule libere: {len(celule_libere)}")
if drum_celule is None:
    print("Nu există drum la această adâncime.")
else:
    puncte = puncte_drum(drum_celule, drum_mijloace)
    lungime = sum(distanta(p, q) for p, q in zip(puncte, puncte[1:]))
    print(f"Drum găsit, prin {len(drum_celule)} celule, "
          f"lungime {lungime:.1f} unități.")

deseneaza_pasi()
deseneaza_rezultat(drum_celule, drum_mijloace)
plt.show()
