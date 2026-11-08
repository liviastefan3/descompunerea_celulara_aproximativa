"""
Descompunere celulară aproximativă - aplicație interactivă
  1. utilizatorul alege câte celule are harta (coloane x rânduri)
  2. desenează obstacole (pătrat, triunghi, cerc) trăgând cu mouse-ul
  3. plasează start și goal
  4. alege metoda: descompunere omogenă (grilă + BFS) sau metoda arborelui (quadtree + A*)

Instalare:  pip install numpy matplotlib shapely
"""
import heapq
import math
import time
import tkinter as tk
from collections import deque
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from shapely import affinity
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep

CELULA = 10          # latura unei celule a grilei omogene (în unități ale hărții)
EPS = 1e-6
LIBER, OCUPAT, MIXT = "liber", "ocupat", "mixt"


# =====================================================================
# Utilitare geometrice
# =====================================================================
def se_suprapune(obs, geom):
    """True dacă `geom` are interiorul comun cu obstacolele.
    Dacă doar se ating pe margine, NU contează ca suprapunere
    (altfel un obstacol desenat pe linii de grilă ar 'crește' cu o celulă)."""
    return obs.intersects(geom) and not obs.touches(geom)


def forma(tip, x0, y0, x1, y1):
    """Creează obstacolul în dreptunghiul (x0,y0)-(x1,y1)."""
    if tip == "patrat":
        return box(x0, y0, x1, y1)
    if tip == "triunghi":
        return Polygon([(x0, y0), (x1, y0), ((x0 + x1) / 2, y1)])
    # cerc / elipsă
    cerc = Point((x0 + x1) / 2, (y0 + y1) / 2).buffer(1, 32)
    return affinity.scale(cerc, (x1 - x0) / 2, (y1 - y0) / 2)


def lungime_drum(puncte):
    return sum(math.hypot(p[0] - q[0], p[1] - q[1])
               for p, q in zip(puncte, puncte[1:]))


# =====================================================================
# METODA 1: descompunere omogenă (grilă + BFS)
# =====================================================================
def construieste_grila(obstacole, n_cols, n_rows):
    grid = np.zeros((n_rows, n_cols), dtype=int)     # 0 = liber, 1 = ocupat
    if not obstacole:
        return grid
    obs = prep(unary_union(obstacole))
    for r in range(n_rows):
        for c in range(n_cols):
            celula = box(c * CELULA, r * CELULA, (c + 1) * CELULA, (r + 1) * CELULA)
            if se_suprapune(obs, celula):
                grid[r][c] = 1
    return grid


def bfs(grid, start, goal):
    n_rows, n_cols = grid.shape
    coada = deque([start])
    parinte = {start: None}
    while coada:
        curent = coada.popleft()
        if curent == goal:
            break
        r, c = curent
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < n_rows and 0 <= nc < n_cols
                    and grid[nr][nc] == 0 and (nr, nc) not in parinte):
                parinte[(nr, nc)] = curent
                coada.append((nr, nc))
    if goal not in parinte:
        return None
    drum, nod = [], goal
    while nod is not None:
        drum.append(nod)
        nod = parinte[nod]
    drum.reverse()
    return drum


def metoda_grila(obstacole, n_cols, n_rows, start_xy, goal_xy):
    t0 = time.perf_counter()
    grid = construieste_grila(obstacole, n_cols, n_rows)
    start = (int(start_xy[1] // CELULA), int(start_xy[0] // CELULA))
    goal = (int(goal_xy[1] // CELULA), int(goal_xy[0] // CELULA))
    if grid[start] == 1 or grid[goal] == 1:
        return {"eroare": "Start sau goal este într-un obstacol!"}
    drum = bfs(grid, start, goal)
    timp = time.perf_counter() - t0
    puncte = None
    if drum:
        puncte = [((c + 0.5) * CELULA, (r + 0.5) * CELULA) for r, c in drum]
    return {"tip": "grila", "grid": grid, "puncte": puncte,
            "n_celule": grid.size, "timp": timp,
            "lungime": lungime_drum(puncte) if puncte else None}


# =====================================================================
# METODA 2: metoda arborelui (quadtree + A*)
# =====================================================================
class Celula:
    def __init__(self, x, y, w, h, adancime, obs):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.adancime = adancime
        self.copii = []
        self.idx = None                      # indexul printre celulele libere
        self.stare = self._clasifica(obs)

    def dreptunghi(self):
        return box(self.x, self.y, self.x + self.w, self.y + self.h)

    def _clasifica(self, obs):
        if obs is None:
            return LIBER
        d = self.dreptunghi()
        if not se_suprapune(obs, d):
            return LIBER
        if obs.contains(d):
            return OCUPAT
        return MIXT

    def centru(self):
        return (self.x + self.w / 2, self.y + self.h / 2)


def construieste_arbore(celula, obs, adancime_max):
    """Împarte recursiv celulele mixte în 4."""
    if celula.stare != MIXT or celula.adancime == adancime_max:
        return
    w2, h2, d = celula.w / 2, celula.h / 2, celula.adancime + 1
    celula.copii = [
        Celula(celula.x, celula.y, w2, h2, d, obs),
        Celula(celula.x + w2, celula.y, w2, h2, d, obs),
        Celula(celula.x, celula.y + h2, w2, h2, d, obs),
        Celula(celula.x + w2, celula.y + h2, w2, h2, d, obs),
    ]
    for copil in celula.copii:
        construieste_arbore(copil, obs, adancime_max)


def frunze(celula):
    if not celula.copii:
        return [celula]
    rezultat = []
    for copil in celula.copii:
        rezultat.extend(frunze(copil))
    return rezultat


def frunza_la(radacina, px, py):
    """Frunza arborelui care conține punctul (px, py), sau None."""
    nod = radacina
    if not (nod.x <= px < nod.x + nod.w and nod.y <= py < nod.y + nod.h):
        return None
    while nod.copii:
        i = (1 if px >= nod.x + nod.w / 2 else 0) + (2 if py >= nod.y + nod.h / 2 else 0)
        nod = nod.copii[i]
    return nod


def latura_comuna(a, b):
    """Mijlocul laturii comune dacă a și b sunt vecine, altfel None."""
    if abs(a.x + a.w - b.x) < EPS or abs(b.x + b.w - a.x) < EPS:
        jos, sus = max(a.y, b.y), min(a.y + a.h, b.y + b.h)
        if sus - jos > EPS:
            x = b.x if abs(a.x + a.w - b.x) < EPS else a.x
            return (x, (jos + sus) / 2)
    if abs(a.y + a.h - b.y) < EPS or abs(b.y + b.h - a.y) < EPS:
        st, dr = max(a.x, b.x), min(a.x + a.w, b.x + b.w)
        if dr - st > EPS:
            y = b.y if abs(a.y + a.h - b.y) < EPS else a.y
            return ((st + dr) / 2, y)
    return None


def construieste_vecini(radacina, libere, adancime_max):
    """Graful de vecinătate: pentru fiecare celulă liberă căutăm vecinii
    pe latura din dreapta și cea de sus (punem puncte-sondă la rezoluția
    cea mai fină și vedem în ce frunză cad)."""
    fw = radacina.w / 2 ** adancime_max
    fh = radacina.h / 2 ** adancime_max
    vecini = [[] for _ in libere]
    vazute = set()
    for a in libere:
        sonde = []
        for k in range(max(1, round(a.h / fh))):          # latura din dreapta
            sonde.append((a.x + a.w + EPS, a.y + fh * (k + 0.5)))
        for k in range(max(1, round(a.w / fw))):          # latura de sus
            sonde.append((a.x + fw * (k + 0.5), a.y + a.h + EPS))
        for px, py in sonde:
            b = frunza_la(radacina, px, py)
            if b is None or b.stare != LIBER or b is a:
                continue
            cheie = (min(a.idx, b.idx), max(a.idx, b.idx))
            if cheie in vazute:
                continue
            vazute.add(cheie)
            m = latura_comuna(a, b)
            if m is not None:
                vecini[a.idx].append((b.idx, m))
                vecini[b.idx].append((a.idx, m))
    return vecini


def distanta(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def a_star(libere, vecini, start, goal, goal_xy):
    def h(i):
        return distanta(libere[i].centru(), goal_xy)

    def cost_pas(i, j, mijloc):
        return distanta(libere[i].centru(), mijloc) + distanta(mijloc, libere[j].centru())

    coada = [(h(start), 0.0, start)]
    cost = {start: 0.0}
    parinte = {start: (None, None)}
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
    celule, mijloace, nod = [], [], goal
    while nod is not None:
        celule.append(nod)
        anterior, mijloc = parinte[nod]
        if mijloc is not None:
            mijloace.append(mijloc)
        nod = anterior
    celule.reverse()
    mijloace.reverse()
    return celule, mijloace


def metoda_arbore(obstacole, n_cols, n_rows, start_xy, goal_xy, adancime_max=6):
    t0 = time.perf_counter()
    W, H = n_cols * CELULA, n_rows * CELULA
    obs = prep(unary_union(obstacole)) if obstacole else None

    radacina = Celula(0, 0, W, H, 0, obs)
    construieste_arbore(radacina, obs, adancime_max)
    toate = frunze(radacina)
    # celulele rămase MIXTE la adâncimea maximă sunt tratate ca ocupate
    libere = [c for c in toate if c.stare == LIBER]
    for i, c in enumerate(libere):
        c.idx = i

    s = frunza_la(radacina, *start_xy)
    g = frunza_la(radacina, *goal_xy)
    if s is None or g is None or s.stare != LIBER or g.stare != LIBER:
        return {"eroare": "Start sau goal este într-un obstacol sau într-o celulă "
                          "mixtă (tratată ca ocupată) la această adâncime. "
                          "Încearcă o adâncime mai mare sau mută punctele."}

    vecini = construieste_vecini(radacina, libere, adancime_max)
    celule, mijloace = a_star(libere, vecini, s.idx, g.idx, goal_xy)
    timp = time.perf_counter() - t0

    puncte = None
    if celule:
        puncte = [start_xy, libere[celule[0]].centru()]
        for mijloc, c in zip(mijloace, celule[1:]):
            puncte += [mijloc, libere[c].centru()]
        puncte.append(goal_xy)
    return {"tip": "quad", "frunze": toate, "libere": libere,
            "celule_drum": celule, "puncte": puncte,
            "n_celule": len(toate), "adancime": adancime_max, "timp": timp,
            "lungime": lungime_drum(puncte) if puncte else None}


# =====================================================================
# INTERFAȚA
# =====================================================================
UNELTE = [
    ("Pătrat / dreptunghi", "patrat"),
    ("Triunghi", "triunghi"),
    ("Cerc / elipsă", "cerc"),
    ("Plasează START", "start"),
    ("Plasează GOAL", "goal"),
    ("Șterge obstacol (click pe el)", "sterge"),
]


class App:
    def __init__(self, root):
        self.root = root
        root.title("Descompunere celulară aproximativă")

        self.n_cols, self.n_rows = 40, 30
        self.obstacole = []              # listă de (tip, geometrie shapely)
        self.rezultat = None
        self.p0 = self.p1 = None         # colțurile dreptunghiului tras cu mouse-ul
        self.previzualizare = None

        self.var_cols = tk.StringVar(value="40")
        self.var_rows = tk.StringVar(value="30")
        self.unealta = tk.StringVar(value="patrat")
        self.metoda = tk.StringVar(value="grila")
        self.var_adancime = tk.StringVar(value="6")
        self.status = tk.StringVar(value="Desenează obstacole, plasează start/goal și apasă „Găsește drum”.")

        self._construieste_ui()
        self.pozitii_implicite()
        self.deseneaza()

    # ---------------- UI ----------------
    def _construieste_ui(self):
        stanga = ttk.Frame(self.root, padding=8)
        stanga.pack(side="left", fill="y")

        f = ttk.LabelFrame(stanga, text="1. Harta", padding=6)
        f.pack(fill="x", pady=4)
        ttk.Label(f, text="Celule în lungime (coloane):").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(f, from_=5, to=80, width=5, textvariable=self.var_cols).grid(row=0, column=1)
        ttk.Label(f, text="Celule în înălțime (rânduri):").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(f, from_=5, to=80, width=5, textvariable=self.var_rows).grid(row=1, column=1)
        ttk.Button(f, text="Generează harta", command=self.genereaza_harta).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(f, text="Încarcă harta exemplu (40x30)", command=self.harta_exemplu).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        f = ttk.LabelFrame(stanga, text="2. Obstacole și puncte", padding=6)
        f.pack(fill="x", pady=4)
        for text, val in UNELTE:
            ttk.Radiobutton(f, text=text, value=val, variable=self.unealta).pack(anchor="w")
        ttk.Label(f, justify="left", wraplength=250, foreground="#555",
                  text="Obstacol: apasă și TRAGE pe hartă (dimensiunea o stabilești "
                       "din cât tragi; se aliniază la celule). Click dreapta pe un "
                       "obstacol îl șterge.").pack(anchor="w", pady=(6, 0))
        ttk.Button(f, text="Șterge toate obstacolele", command=self.sterge_tot).pack(
            fill="x", pady=(6, 0))

        f = ttk.LabelFrame(stanga, text="3. Metoda", padding=6)
        f.pack(fill="x", pady=4)
        ttk.Radiobutton(f, text="Descompunere omogenă (grilă + BFS)",
                        value="grila", variable=self.metoda).pack(anchor="w")
        ttk.Radiobutton(f, text="Metoda arborelui (quadtree + A*)",
                        value="arbore", variable=self.metoda).pack(anchor="w")
        fa = ttk.Frame(f)
        fa.pack(anchor="w", pady=(6, 0))
        ttk.Label(fa, text="Adâncime arbore (1-6):").pack(side="left")
        ttk.Spinbox(fa, from_=1, to=6, width=4, state="readonly",
                    textvariable=self.var_adancime).pack(side="left", padx=4)
        ttk.Button(f, text="Găsește drum", command=self.ruleaza).pack(fill="x", pady=(8, 0))
        ttk.Button(f, text="Șterge drumul / revino la editare",
                   command=self.sterge_rezultat).pack(fill="x", pady=(4, 0))

        ttk.Label(stanga, textvariable=self.status, wraplength=270, justify="left",
                  relief="sunken", padding=6).pack(fill="x", pady=8)

        dreapta = ttk.Frame(self.root)
        dreapta.pack(side="right", fill="both", expand=True)
        self.fig = Figure(figsize=(8.5, 6.5))
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
        self.canvas = FigureCanvasTkAgg(self.fig, master=dreapta)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self.la_apasare)
        self.canvas.mpl_connect("motion_notify_event", self.la_miscare)
        self.canvas.mpl_connect("button_release_event", self.la_eliberare)

    # ---------------- stare ----------------
    @property
    def W(self):
        return self.n_cols * CELULA

    @property
    def H(self):
        return self.n_rows * CELULA

    def pozitii_implicite(self):
        self.start_xy = (1.5 * CELULA, 1.5 * CELULA)
        self.goal_xy = ((self.n_cols - 1.5) * CELULA, (self.n_rows - 1.5) * CELULA)

    def genereaza_harta(self):
        try:
            c, r = int(self.var_cols.get()), int(self.var_rows.get())
        except ValueError:
            messagebox.showerror("Eroare", "Introdu numere întregi pentru coloane și rânduri.")
            return
        if not (5 <= c <= 80 and 5 <= r <= 80):
            messagebox.showerror("Eroare", "Numărul de celule trebuie să fie între 5 și 80.")
            return
        self.n_cols, self.n_rows = c, r
        self.obstacole = []
        self.rezultat = None
        self.pozitii_implicite()
        self.status.set(f"Hartă nouă: {c} x {r} celule.")
        self.deseneaza()

    def harta_exemplu(self):
        self.var_cols.set("40")
        self.var_rows.set("30")
        self.n_cols, self.n_rows = 40, 30
        self.obstacole = [
            ("patrat", Polygon([(100, 200), (300, 200), (300, 240), (100, 240)])),
            ("patrat", Polygon([(250, 40), (280, 40), (280, 200), (250, 200)])),
            ("triunghi", Polygon([(60, 70), (180, 70), (160, 120)])),
            ("triunghi", Polygon([(300, 20), (320, 20), (310, 150)])),
        ]
        self.start_xy = (25, 105)
        self.goal_xy = (375, 25)
        self.rezultat = None
        self.status.set("Harta exemplu încărcată.")
        self.deseneaza()

    def sterge_tot(self):
        self.obstacole = []
        self.rezultat = None
        self.deseneaza()

    def sterge_rezultat(self):
        self.rezultat = None
        self.status.set("Mod editare.")
        self.deseneaza()

    # ---------------- mouse ----------------
    def snap(self, x, y):
        sx = min(max(round(x / CELULA) * CELULA, 0), self.W)
        sy = min(max(round(y / CELULA) * CELULA, 0), self.H)
        return (sx, sy)

    def centru_celulei(self, x, y):
        c = min(max(int(x // CELULA), 0), self.n_cols - 1)
        r = min(max(int(y // CELULA), 0), self.n_rows - 1)
        return ((c + 0.5) * CELULA, (r + 0.5) * CELULA)

    def sterge_la(self, x, y):
        p = Point(x, y)
        inainte = len(self.obstacole)
        self.obstacole = [o for o in self.obstacole if not o[1].contains(p)]
        if len(self.obstacole) != inainte:
            self.rezultat = None
            self.deseneaza()

    def la_apasare(self, e):
        if e.inaxes is not self.ax or e.xdata is None:
            return
        if e.button == 3:                       # click dreapta = șterge
            self.sterge_la(e.xdata, e.ydata)
            return
        if e.button != 1:
            return
        u = self.unealta.get()
        if u == "start":
            self.start_xy = self.centru_celulei(e.xdata, e.ydata)
            self.rezultat = None
            self.deseneaza()
        elif u == "goal":
            self.goal_xy = self.centru_celulei(e.xdata, e.ydata)
            self.rezultat = None
            self.deseneaza()
        elif u == "sterge":
            self.sterge_la(e.xdata, e.ydata)
        else:
            if self.rezultat is not None:       # edităm => ieșim din modul rezultat
                self.rezultat = None
                self.deseneaza()
            self.p0 = self.p1 = self.snap(e.xdata, e.ydata)
            self.previzualizare = MplPolygon([[0, 0]], closed=True, facecolor="blue",
                                             alpha=0.35, edgecolor="navy", zorder=6)
            self.ax.add_patch(self.previzualizare)

    def la_miscare(self, e):
        if self.p0 is None or e.inaxes is not self.ax or e.xdata is None:
            return
        self.p1 = self.snap(e.xdata, e.ydata)
        x0, x1 = sorted((self.p0[0], self.p1[0]))
        y0, y1 = sorted((self.p0[1], self.p1[1]))
        if x1 - x0 > 0 and y1 - y0 > 0:
            g = forma(self.unealta.get(), x0, y0, x1, y1)
            self.previzualizare.set_xy(list(g.exterior.coords))
        else:
            self.previzualizare.set_xy([[0, 0]])
        self.canvas.draw_idle()

    def la_eliberare(self, e):
        if self.p0 is None:
            return
        if e.inaxes is self.ax and e.xdata is not None:
            self.p1 = self.snap(e.xdata, e.ydata)
        x0, x1 = sorted((self.p0[0], self.p1[0]))
        y0, y1 = sorted((self.p0[1], self.p1[1]))
        tip = self.unealta.get()
        self.p0 = self.p1 = None
        self.previzualizare = None
        if x1 - x0 >= CELULA and y1 - y0 >= CELULA:
            self.obstacole.append((tip, forma(tip, x0, y0, x1, y1)))
            self.rezultat = None
        self.deseneaza()

    # ---------------- rulare ----------------
    def ruleaza(self):
        obs = [g for _, g in self.obstacole]
        nume = "Descompunere omogenă (BFS)" if self.metoda.get() == "grila" \
            else "Metoda arborelui (A*)"
        self.status.set("Se calculează...")
        self.root.update_idletasks()
        if self.metoda.get() == "grila":
            r = metoda_grila(obs, self.n_cols, self.n_rows, self.start_xy, self.goal_xy)
        else:
            adancime = int(self.var_adancime.get())
            r = metoda_arbore(obs, self.n_cols, self.n_rows, self.start_xy,
                              self.goal_xy, adancime)
        if "eroare" in r:
            self.status.set(r["eroare"])
            messagebox.showerror("Eroare", r["eroare"])
            return
        self.rezultat = r
        if r["tip"] == "quad":
            nume += f", adâncime {r['adancime']}"
        text = f"{nume}\nCelule folosite: {r['n_celule']}\nTimp: {r['timp'] * 1000:.1f} ms\n"
        if r["puncte"] is None:
            text += "Nu există drum la această rezoluție."
        else:
            text += f"Drum găsit, lungime ≈ {r['lungime']:.1f} unități."
        self.status.set(text)
        self.deseneaza()

    # ---------------- desen ----------------
    def _linii_grila(self):
        ax = self.ax
        ax.set_xticks(np.arange(0, self.W + 1, CELULA))
        ax.set_yticks(np.arange(0, self.H + 1, CELULA))
        ax.grid(True, color="lightgray", linewidth=0.5)
        ax.tick_params(labelbottom=False, labelleft=False, length=0)

    def deseneaza(self):
        ax = self.ax
        ax.clear()
        r = self.rezultat
        if r and r["tip"] == "grila":
            ax.imshow(r["grid"], cmap="Greys", vmin=0, vmax=1, origin="lower",
                      extent=(0, self.W, 0, self.H))
            self._linii_grila()
            ax.grid(True, color="gray", linewidth=0.5)
        elif r and r["tip"] == "quad":
            patch_uri, culori = [], []
            drum = set(r["celule_drum"] or [])
            for c in r["frunze"]:
                patch_uri.append(Rectangle((c.x, c.y), c.w, c.h))
                if c.stare != LIBER:
                    culori.append("#9e9e9e")        # ocupat (sau mixt la adâncimea maximă)
                elif c.idx in drum:
                    culori.append("#ffe0b2")        # celule prin care trece drumul
                else:
                    culori.append("white")
            ax.add_collection(PatchCollection(patch_uri, facecolors=culori,
                                              edgecolors="red", linewidths=0.4))
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            self._linii_grila()

        for _, g in self.obstacole:
            xs, ys = g.exterior.xy
            ax.fill(xs, ys, color="blue", alpha=0.6, zorder=3)

        if r and r.get("puncte"):
            xs, ys = zip(*r["puncte"])
            ax.plot(xs, ys, color="orange", linewidth=2.5, zorder=5)

        ax.plot(*self.start_xy, "bo", markersize=10, zorder=7, markeredgecolor="white")
        ax.plot(*self.goal_xy, "ro", markersize=10, zorder=7, markeredgecolor="white")

        ax.set_xlim(0, self.W)
        ax.set_ylim(0, self.H)
        ax.set_aspect("equal")
        if r is None:
            titlu = f"Editare: {self.n_cols} x {self.n_rows} celule"
        elif r["tip"] == "grila":
            titlu = "Descompunere omogenă"
        else:
            titlu = f"Metoda arborelui (adâncime maximă {r['adancime']})"
        ax.set_title(titlu)
        self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()