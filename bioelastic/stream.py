"""Flujo continuo largo y no estacionario para medir aprendizaje continuo.

El flujo es una sucesión de fases. Cada fase fija qué clases aparecen, cómo llega la
enseñanza y en qué contexto llegan las entradas:

  modo      "sup"     con etiqueta (prueba y después enseñanza)
            "reward"  sólo recompensa escalar: la red elige y recibe +1 / -1
            "unlab"   sin etiquetas ni recompensa (sólo observar)
  contexto  "clean"   la imagen tal cual
            "noise"   ruido gaussiano y sal (entrada ruidosa)
            "occlude" un bloque tapado y píxeles perdidos (entrada incompleta)
            "rot90"   imagen rotada: las mismas clases con otra apariencia

El calendario por defecto (`SCHEDULE`, 200 000 pasos con `scale=1`) mezcla MNIST
(clases 0-9) y Fashion-MNIST (clases 10-19) en cinco grupos. Las clases aparecen,
desaparecen y vuelven después de intervalos largos; hay fases sólo con recompensa,
fases sin etiquetas, contextos que cambian y vuelven, y al final un sondeo con dos
clases que nunca se vieron (plasticidad tardía).

Todo es determinista dada la semilla: el mismo flujo puede alimentar a la red
entrenada y a una red nueva (para comparar plasticidad al principio y al final).

Métricas (funciones puras sobre los registros, ver tests):
  forgetting   olvido medio por grupo: máximo previo - final (Chaudhry et al. 2018)
  window_acc   acierto prequential en los primeros pasos de una fase (ahorro al reaparecer)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GROUPS = {
    "A": (0, 1, 2, 3, 4),          # MNIST 0-4
    "B": (5, 6, 7, 8, 9),          # MNIST 5-9
    "C": (10, 11, 12, 13),         # Fashion: remera, pantalón, pulóver, vestido
    "D": (14, 15, 16, 17),         # Fashion: abrigo, sandalia, camisa, zapatilla
    "P": (18, 19),                 # Fashion: cartera, bota (sólo en el sondeo final)
}
KNOWN = GROUPS["A"] + GROUPS["B"] + GROUPS["C"] + GROUPS["D"]


@dataclass(frozen=True)
class Phase:
    name: str
    groups: tuple[str, ...]
    steps: int                      # pasos con scale=1
    mode: str = "sup"
    ctx: str = "clean"
    mix: tuple[str, ...] = ()       # grupos ya conocidos que se intercalan
    mix_frac: float = 0.0           # fracción de pasos de `mix`

    @property
    def classes(self) -> tuple[int, ...]:
        return tuple(c for g in self.groups for c in GROUPS[g])

    @property
    def mix_classes(self) -> tuple[int, ...]:
        return tuple(c for g in self.mix for c in GROUPS[g])


SCHEDULE = (
    Phase("A", ("A",), 16000),
    Phase("B", ("B",), 16000),
    Phase("C", ("C",), 16000),
    Phase("A_reward", ("A",), 12000, mode="reward"),                  # A vuelve, sólo recompensa
    Phase("D", ("D",), 16000),
    Phase("BC_unlab_noise", ("B", "C"), 16000, mode="unlab", ctx="noise"),
    Phase("AB_rot", ("A", "B"), 20000, ctx="rot90"),                  # contexto nuevo
    Phase("CD_occl", ("C", "D"), 16000, ctx="occlude"),               # entradas incompletas
    Phase("B_back", ("B",), 12000),                                   # B vuelve tras ~110k pasos
    Phase("all_reward_noise", ("A", "B", "C", "D"), 24000, mode="reward", ctx="noise"),
    Phase("AB_rot_back", ("A", "B"), 12000, ctx="rot90"),             # el contexto vuelve
    Phase("D_back", ("D",), 12000),
    Phase("P_probe", ("P",), 12000, mix=("A", "B", "C", "D"), mix_frac=0.5),
)


# ---------------------------------------------------------------------- contextos
def transform(img: np.ndarray, ctx: str, rng: np.random.Generator) -> np.ndarray:
    """Imagen (28, 28) en [0, 1] -> imagen del contexto `ctx` en [0, 1]."""
    x = np.asarray(img, dtype=np.float32)
    if ctx == "clean":
        return x
    if ctx == "noise":
        x = x + rng.normal(0.0, 0.2, x.shape).astype(np.float32)
        salt = rng.random(x.shape) < 0.03
        x[salt] = rng.uniform(0.5, 1.0, int(salt.sum()))
        return np.clip(x, 0.0, 1.0)
    if ctx == "occlude":
        x = x.copy()
        H, W = x.shape
        s = H // 2 - 2                                   # bloque de 12x12 en una posición al azar
        y0, x0 = (int(v) for v in rng.integers(0, H - s + 1, size=2))
        x[y0:y0 + s, x0:x0 + s] = 0.0
        x[rng.random(x.shape) < 0.2] = 0.0               # y un 20 % de píxeles perdidos
        return x
    if ctx == "rot90":
        return np.ascontiguousarray(np.rot90(x))
    raise ValueError(f"contexto desconocido: {ctx}")


# ---------------------------------------------------------------------- flujo
@dataclass
class Event:
    t: int
    phase: int
    x: np.ndarray                   # imagen del contexto, (28, 28) en [0, 1]
    y: int
    mode: str
    ctx: str
    new: bool                       # True si la clase es de los grupos de la fase (no de `mix`)


@dataclass
class ContinualStream:
    """Flujo determinista sobre bancos de imágenes por clase.

    pools : {clase: array (n, 28, 28) uint8 o float en [0, 1]}
    scale : factor sobre la duración de cada fase (1.0 = calendario completo)
    """
    pools: dict
    schedule: tuple = SCHEDULE
    scale: float = 1.0
    seed: int = 0
    _ptr: dict = field(default_factory=dict, init=False, repr=False)
    _perm: dict = field(default_factory=dict, init=False, repr=False)

    def phase_steps(self) -> list[int]:
        return [max(1, int(round(p.steps * self.scale))) for p in self.schedule]

    def __len__(self) -> int:
        return int(sum(self.phase_steps()))

    def bounds(self) -> list[tuple[int, int]]:
        """[(inicio, fin)) de cada fase en pasos absolutos."""
        out, t = [], 0
        for n in self.phase_steps():
            out.append((t, t + n)); t += n
        return out

    def _draw(self, c: int, rng) -> np.ndarray:
        """Siguiente imagen de la clase c: recorre permutaciones sin reemplazo."""
        pool = self.pools[c]
        if c not in self._perm or self._ptr[c] >= len(pool):
            self._perm[c] = rng.permutation(len(pool)); self._ptr[c] = 0
        i = int(self._perm[c][self._ptr[c]]); self._ptr[c] += 1
        img = pool[i]
        return img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)

    def phase_events(self, k: int, rng_seed: int | None = None):
        """Eventos de una sola fase (para reproducir el inicio de una fase en una red nueva)."""
        rng = np.random.default_rng([self.seed if rng_seed is None else rng_seed, 7919, k])
        ph = self.schedule[k]
        t0 = self.bounds()[k][0]
        own = ph.classes; mix = ph.mix_classes
        for i in range(self.phase_steps()[k]):
            new = not (mix and rng.random() < ph.mix_frac)
            c = int(rng.choice(own)) if new else int(rng.choice(mix))
            x = transform(self._draw(c, rng), ph.ctx, rng)
            yield Event(t0 + i, k, x, c, ph.mode, ph.ctx, new)

    def __iter__(self):
        self._ptr.clear(); self._perm.clear()
        for k in range(len(self.schedule)):
            yield from self.phase_events(k)


def first_appearance(schedule=SCHEDULE) -> dict:
    """{grupo: índice de la fase donde aparece por primera vez}."""
    out = {}
    for k, ph in enumerate(schedule):
        for g in ph.groups:
            out.setdefault(g, k)
    return out


def reappearances(schedule=SCHEDULE, mode: str = "sup") -> list[tuple[int, int, str]]:
    """[(fase anterior, fase de regreso, contexto)] para grupos que vuelven en `mode`
    con el mismo contexto después de haber estado ausentes al menos una fase."""
    out = []
    last = {}
    for k, ph in enumerate(schedule):
        if ph.mode == mode:
            key = (ph.groups, ph.ctx)
            if key in last and last[key] < k - 1:
                out.append((last[key], k, ph.ctx))
            last[key] = k
    return out


# ---------------------------------------------------------------------- métricas
def forgetting(history: list[dict], final: dict | None = None) -> dict:
    """Olvido por clave: máximo de la precisión en los puntos previos menos la final.

    history : lista de {clave: precisión} en orden temporal (claves ausentes = no medidas)
    final   : precisión final (por defecto, el último punto de `history`)
    Sólo cuenta claves medidas antes del final con algún valor > 0 (ya aprendidas).
    """
    final = history[-1] if final is None else final
    prev = history[:-1] if final is history[-1] else history
    out = {}
    for key, acc in final.items():
        past = [h[key] for h in prev if key in h]
        if past and max(past) > 0:
            out[key] = float(max(past) - acc)
    return out


def window_acc(hits: np.ndarray, start: int, n: int) -> float:
    """Acierto medio en hits[start:start+n]."""
    seg = np.asarray(hits[start:start + n], dtype=np.float64)
    return float(seg.mean()) if len(seg) else float("nan")


def steps_to(hits: np.ndarray, start: int, end: int, level: float, win: int) -> int:
    """Pasos desde `start` hasta que el acierto en una ventana de `win` alcanza `level`
    (o `end - start` si no llega)."""
    seg = np.asarray(hits[start:end], dtype=np.float64)
    if len(seg) < win:
        return end - start
    run = np.convolve(seg, np.ones(win) / win, mode="valid")
    ok = np.flatnonzero(run >= level - 1e-9)
    return int(ok[0] + win) if len(ok) else end - start
