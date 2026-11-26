"""Aprendizaje continuo: flujo largo y no estacionario (MNIST + Fashion-MNIST).

Ver `bioelastic/stream.py` para el calendario (200 000 pasos con --scale 1). Cada
corrida guarda un JSON en results/continual/ con las curvas y las métricas:

  olvido          por grupo y contexto: máximo previo - final (evaluación en test)
  ahorro          acierto en los primeros pasos cuando un grupo vuelve vs su primera vez
  transferencia   acierto en los primeros pasos de un grupo nuevo vs una red nueva
  plasticidad     sondeo final con dos clases nunca vistas: red entrenada vs red nueva
  crecimiento     neuronas, rasgos, sinapsis y bytes a lo largo del flujo

  python experiments/exp_continual.py --models d1 d2 --seeds 0 1 2 --scale 0.1
  python experiments/exp_continual.py --summary          # agrega los JSON en un CSV
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import RESULTS, CsvLog
from bioelastic import Cortex
from bioelastic.datasets import load_fashion, load_mnist
from bioelastic.stream import (GROUPS, KNOWN, SCHEDULE, ContinualStream, first_appearance,
                               forgetting, reappearances, steps_to, transform, window_acc)
from exp_vision import V1, encoder

N_CLASSES = 20
OUT = RESULTS / "continual"
EVAL_SETS = {                       # contexto -> grupos evaluados
    "clean": ("A", "B", "C", "D", "P"),
    "rot90": ("A", "B"),
    "occlude": ("C", "D"),
    "noise": ("A", "B", "C", "D"),
}


# ====================================================================== modelos
class CortexAgent:
    """Adaptador de `Cortex` al protocolo del flujo (sup / reward / unlab / scores)."""

    def __init__(self, seed: int = 0, **cfg):
        self.net = Cortex((28, 28, 1), N_CLASSES, seed=seed, **cfg)

    def sup(self, s, y):
        return self.net.step(s, y)

    def reward(self, s, y):
        a = self.net.forward(s, explore=True)
        self.net.reinforce(1.0 if a == y else -1.0)
        return a

    def unlab(self, s, y):
        self.net.observe(s)
        return self.net.pred

    def scores(self, s):
        self.net.forward(s)
        return self.net._V.copy()

    def last_scores(self):
        return self.net._V

    def stats(self) -> dict:
        st = self.net.stats()
        nb = sum(h.layer.nbytes() + h.layer.n * N_CLASSES * 4 for h in self.net.heads)
        nb += sum(int(a.alive.sum()) * a.D * 4 for a in self.net.areas)
        if self.net.proj is not None:
            nb += self.net.proj.n * N_CLASSES * 4
        return dict(neurons=st["neurons"], features=st["features"], synapses=st["synapses"],
                    bytes=int(nb), replays=st["replays"],
                    evicted=sum(h.layer.evicted for h in self.net.heads))

    def on_phase(self, k: int):
        pass


class NoNeurogenesis(CortexAgent):
    """Ablación: la neurogénesis (capa y áreas) se apaga al terminar la primera fase."""

    def on_phase(self, k: int):
        if k == 1:
            for h in self.net.heads:
                h.layer.neurogenesis = False
            for a in self.net.areas:
                a.neurogenesis = False


class BalancedBuffer:
    """Memoria de ejemplos con cupo parejo por clase (muestreador de GDumb, Prabhu et al. 2020)."""

    def __init__(self, size: int, dim: int, seed: int = 0):
        self.X = np.zeros((size, dim), dtype=np.float32)
        self.y = np.full(size, -1, dtype=np.int64)
        self.n = 0
        self.rng = np.random.default_rng(seed)

    def add(self, x, y):
        if self.n < len(self.X):
            self.X[self.n] = x; self.y[self.n] = y; self.n += 1
            return
        cnt = np.bincount(self.y, minlength=N_CLASSES)
        big = int(np.argmax(cnt))
        if cnt[y] >= cnt[big]:
            return
        i = int(self.rng.choice(np.flatnonzero(self.y == big)))
        self.X[i] = x; self.y[i] = y

    def sample(self, m):
        if self.n == 0:
            return self.X[:0], self.y[:0]
        i = self.rng.integers(0, self.n, size=m)
        return self.X[i], self.y[i]


class _Baseline:
    """Protocolo común de las referencias: aprenden de la etiqueta y, con recompensa,
    sólo de los aciertos (la acción premiada es la etiqueta). Sin etiquetas no aprenden."""
    eps = 0.1

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.seen = np.zeros(N_CLASSES, dtype=bool)

    def last_scores(self):
        return self._last

    def _masked(self, z):
        z = np.asarray(z, dtype=np.float64).copy()
        if self.seen.any():
            z[~self.seen] = -np.inf
        return z

    def sup(self, s, y):
        x = s.ravel()
        self._last = self.scores(s)
        pred = int(np.argmax(self._last))
        self.seen[y] = True
        self.learn(x, y)
        return pred

    def reward(self, s, y):
        if not self.seen.any() or self.rng.random() < self.eps:
            pool = np.flatnonzero(self.seen) if self.seen.any() else np.arange(N_CLASSES)
            a = int(self.rng.choice(pool))
        else:
            a = int(np.argmax(self.scores(s)))
        self._last = self.scores(s)
        if a == y:
            self.seen[y] = True
            self.learn(s.ravel(), y)
        return a

    def unlab(self, s, y):
        self._last = self.scores(s)
        return int(np.argmax(self._last))

    def on_phase(self, k):
        pass


class KNNBuffer(_Baseline):
    """1-vecino más cercano (coseno) sobre una memoria balanceada de ejemplos."""

    def __init__(self, seed=0, size=2000):
        super().__init__(seed)
        self.buf = BalancedBuffer(size, 784, seed)
        self._nrm = np.zeros(size, dtype=np.float32)

    def learn(self, x, y):
        self.buf.add(x, y)
        self._nrm[:self.buf.n] = np.linalg.norm(self.buf.X[:self.buf.n], axis=1) + 1e-6

    def scores(self, s):
        z = np.zeros(N_CLASSES)
        n = self.buf.n
        if n == 0:
            return z
        x = s.ravel()
        sim = (self.buf.X[:n] @ x) / (self._nrm[:n] * (np.linalg.norm(x) + 1e-6))
        np.maximum.at(z, self.buf.y[:n], sim + 1.0)
        return self._masked(np.where(z > 0, z, -1.0))

    def stats(self):
        return dict(neurons=self.buf.n, features=0, synapses=self.buf.n * 784, bytes=self.buf.X.nbytes)


class StreamingLDA(_Baseline):
    """LDA en flujo (Hayes & Kanan 2020): medias por clase y covarianza compartida online,
    con contracción hacia la identidad. La precisión se recalcula cada `every` pasos y
    cada vez que aparece una clase nueva."""

    def __init__(self, seed=0, shrink=1e-2, every=200, size=None):
        super().__init__(seed)
        D = 784
        self.mu = np.zeros((N_CLASSES, D)); self.cnt = np.zeros(N_CLASSES)
        self.S = np.zeros((D, D)); self.n = 0
        self.shrink = shrink; self.every = every; self._since = every; self._W = None

    def learn(self, x, y):
        x = x.astype(np.float64)
        if self.cnt[y] == 0:
            self._since = self.every                          # clase nueva: recalcular ya
        d = x - self.mu[y]
        self.S = (self.n * self.S + np.outer(d, d) * (self.cnt[y] / (self.cnt[y] + 1.0))) / (self.n + 1.0)
        self.mu[y] += d / (self.cnt[y] + 1.0)
        self.cnt[y] += 1; self.n += 1; self._since += 1

    def scores(self, s):
        if self._W is None or self._since >= self.every:
            P = np.linalg.inv((1.0 - self.shrink) * self.S + self.shrink * np.eye(len(self.S)))
            self._W = P @ self.mu.T
            self._b = -0.5 * np.einsum("cd,dc->c", self.mu, self._W)
            self._since = 0
        z = s.ravel() @ self._W + self._b
        return self._masked(z)

    def stats(self):
        return dict(neurons=int((self.cnt > 0).sum()), features=0, synapses=self.S.size,
                    bytes=int((self.S.size + self.mu.size) * 4))


class MLPReplay(_Baseline):
    """MLP 784-256-20 con SGD online y repaso de experiencia (ER, Chaudhry et al. 2019):
    cada paso entrena con el ejemplo actual y `batch - 1` ejemplos de la memoria balanceada."""

    def __init__(self, seed=0, hidden=256, lr=0.05, size=2000, batch=10):
        super().__init__(seed)
        r = self.rng
        self.W1 = r.normal(0, np.sqrt(2 / 784), (784, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = r.normal(0, np.sqrt(1 / hidden), (hidden, N_CLASSES)).astype(np.float32)
        self.b2 = np.zeros(N_CLASSES, dtype=np.float32)
        self.lr = lr; self.batch = batch
        self.buf = BalancedBuffer(size, 784, seed)

    def scores(self, s):
        h = np.maximum(s.ravel() @ self.W1 + self.b1, 0.0)
        return self._masked(h @ self.W2 + self.b2)

    def learn(self, x, y):
        Xb, yb = self.buf.sample(self.batch - 1)
        X = np.vstack([x[None], Xb]); Y = np.concatenate([[y], yb])
        H = np.maximum(X @ self.W1 + self.b1, 0.0)
        Z = H @ self.W2 + self.b2
        Z[:, ~self.seen] = -1e9
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(Y)), Y] -= 1.0
        G = P / len(Y)
        gW2 = H.T @ G; gH = (G @ self.W2.T) * (H > 0)
        self.W2 -= self.lr * gW2; self.b2 -= self.lr * G.sum(0)
        self.W1 -= self.lr * (X.T @ gH); self.b1 -= self.lr * gH.sum(0)
        self.buf.add(x, y)

    def stats(self):
        nb = self.buf.X.nbytes + (self.W1.size + self.W2.size) * 4
        return dict(neurons=self.W1.shape[1], features=0, synapses=self.W1.size + self.W2.size, bytes=nb)


D2 = dict(areas=[V1], heads=[dict(source="input")], projection=dict(source="all"),
          replay=0.5, joint=False)


def _with(cfg, *, head=None, layer=None, readout=None, area=None, **top):
    """Copia de `cfg` con cambios en el primer head (y su capa/lectura) y en las áreas."""
    cfg = json.loads(json.dumps(cfg))
    cfg.update(top)
    h = cfg["heads"][0]
    h.update(head or {})
    if layer:
        h["layer"] = {**h.get("layer", {}), **layer}
    if readout:
        h["readout"] = {**h.get("readout", {}), **readout}
    if area:
        cfg["areas"] = [{**a, **area} for a in cfg.get("areas", [])]
    return cfg


NEVER = 10 ** 12
D1 = dict(heads=[{}])
BASES = {"d1": D1, "d2": D2}
BASELINES = {"knn_buffer": KNNBuffer, "slda": StreamingLDA, "mlp_replay": MLPReplay}


def build(spec: str):
    """Fábrica(seed) a partir de un nombre «base+mod+mod=valor».

    base: d1 (un área sobre los píxeles) | d2 (V1 + memoria + proyección + repaso) |
          knn_buffer | slda | mlp_replay
    mods: budget=N     presupuesto de neuronas de asociación (desalojo)
          no_neuro     neurogénesis apagada después de la primera fase
          no_prune     sin poda de período crítico (capa y áreas)
          no_replay    sin repaso
          no_consol    sin consolidación de la lectura
          top          la memoria lee de V1 en vez de los píxeles (el repaso queda sin fuente)
          buffer=N     tamaño de la memoria de ejemplos de las referencias
          gate=N       error asimétrico: clases sin objetivo en N aprendizajes, inactivas
          protect      además, no tocar neuronas de clases inactivas
          merge        desalojo por fusión del par más redundante (en vez de utilidad)
          relax=N      metaplasticidad de la lectura que se relaja con constante N
          v1=N         tamaño máximo del diccionario de V1
          anti=X       fuerza del anti-STDP
          utau=N       constante de tiempo de la utilidad (desalojo por utilidad)
          pgate=N      error asimétrico en la proyección directa (ER-ACE local)
          pnorm=X      escalado sináptico de las salidas de la proyección (norma X)
          pmean        la proyección aprende medias de clase (regla hebbiana sin término negativo)
          rehearse=X   ensayos por paso en la propia memoria (Cortex._rehearse_one)
          replay=X     repasos por paso
          vote=X       voto por orden de llegada con constante X (ver BioElasticNet._votes)
          k=N          ganadoras por volea en la memoria
          tc=X         comprime las constantes de tiempo (poda, utilidad) por X, para
                       explorar con --scale X manteniendo las proporciones del flujo
    """
    base, *mods = spec.split("+")
    kv = dict((m.split("=") + [None])[:2] for m in mods)
    if base in BASELINES:
        size = int(kv.get("buffer") or 2000)
        cls = BASELINES[base]
        return (lambda s: cls(s, size=size)) if base != "slda" else (lambda s: cls(s))
    cfg = BASES[base]
    layer, readout, area, top, head = {}, {}, {}, {}, {}
    agent = CortexAgent
    for m, v in kv.items():
        if m == "budget":
            layer["max_neurons"] = int(v)
        elif m == "no_neuro":
            agent = NoNeurogenesis
        elif m == "no_prune":
            layer["probation"] = NEVER; area["probation"] = NEVER
        elif m == "no_replay":
            top["replay"] = 0.0
        elif m == "no_consol":
            readout["consolidation"] = 0.0
        elif m == "top":
            head["source"] = "top"
        elif m == "gate":
            readout["active_window"] = int(v)
        elif m == "protect":
            head["protect"] = True
        elif m == "merge":
            layer["evict"] = "merge"
        elif m == "relax":
            readout["relax"] = float(v)
        elif m == "v1":
            area["max_features"] = int(v)
        elif m == "anti":
            head["anti"] = float(v)
        elif m == "utau":
            layer["util_tau"] = float(v)
        elif m == "pgate":
            top["projection"] = {**top.get("projection", cfg.get("projection", {})), "active_window": int(v)}
        elif m == "pnorm":
            top["projection"] = {**top.get("projection", cfg.get("projection", {})), "row_norm": float(v)}
        elif m == "pmean":
            top["projection"] = {**top.get("projection", cfg.get("projection", {})), "rule": "mean"}
        elif m == "rehearse":
            top["rehearse"] = float(v)
        elif m == "replay":
            top["replay"] = float(v)
        elif m == "vote":
            head["vote_tau"] = float(v)
        elif m == "k":
            layer["k"] = int(v)
        elif m == "tc":
            x = float(v)
            layer.setdefault("probation", int(20000 * x)); layer.setdefault("util_tau", 20000.0 * x)
            area.setdefault("probation", int(2000 * x))
            head["maintain_every"] = max(50, int(500 * x))
        else:
            raise ValueError(f"modificador desconocido: {m}")
    cfg = _with(cfg, head=head, layer=layer, readout=readout, area=area, **top)
    return lambda s: agent(s, **cfg)


# ====================================================================== evaluación
def make_eval(Xte_by_class, enc, n_per_class: int, seed: int) -> dict:
    """{contexto: (S codificadas, etiquetas)} fijo para la corrida."""
    rng = np.random.default_rng([seed, 31337])
    out = {}
    for ctx, groups in EVAL_SETS.items():
        S, Y = [], []
        for g in groups:
            for c in GROUPS[g]:
                pool = Xte_by_class[c]
                for i in rng.choice(len(pool), n_per_class, replace=False):
                    S.append(enc(transform(pool[i].astype(np.float32) / 255.0, ctx, rng))); Y.append(c)
        out[ctx] = (S, np.array(Y))
    return out


def evaluate(agent, evals) -> dict:
    """Precisión por contexto y por clase, y la de los grupos (argmax sobre las 20 salidas).
    Para el grupo del sondeo también la «restringida» (argmax sólo entre sus clases)."""
    out = {}
    for ctx, (S, Y) in evals.items():
        Z = np.stack([agent.scores(s) for s in S])
        pred = Z.argmax(axis=1)
        for g in EVAL_SETS[ctx]:
            m = np.isin(Y, GROUPS[g])
            out[f"{ctx}/{g}"] = float(np.mean(pred[m] == Y[m]))
            if g == "P":
                cls = np.array(GROUPS[g])
                out[f"{ctx}/{g}_restricted"] = float(np.mean(cls[Z[m][:, cls].argmax(axis=1)] == Y[m]))
        if ctx == "clean":
            for c in np.unique(Y):
                out[f"clean/c{c}"] = float(np.mean(pred[Y == c] == c))
        known = np.isin(Y, KNOWN)
        out[f"{ctx}/known"] = float(np.mean(pred[known] == Y[known]))
    return out


# ====================================================================== corrida
def run(name: str, seed: int, scale: float, n_eval: int, data) -> dict:
    factory = build(name)
    pools, test_pools = data
    enc = encoder()
    stream = ContinualStream(pools, scale=scale, seed=seed)
    bounds = stream.bounds()
    T = len(stream)
    W = max(200, int(round(1000 * scale)))                 # ventana de "primeros pasos"
    evals = make_eval(test_pools, enc, n_eval, seed)
    first = first_appearance()
    keep_from = {first[g]: g for g in ("B", "C", "D")}     # transferencia: inicio de grupos nuevos
    probe_k = len(SCHEDULE) - 1
    saved = {g: [] for g in keep_from.values()}; saved["P"] = []

    agent = factory(seed)
    hits = np.zeros(T, dtype=np.int8); hits_r = np.zeros(T, dtype=np.int8); ys = np.zeros(T, dtype=np.int8); newf = np.zeros(T, dtype=bool)
    growth = []; checkpoints = []
    log_every = max(500, T // 200)
    t0 = time.time()
    phase = -1
    for ev in stream:
        if ev.phase != phase:
            phase = ev.phase
            agent.on_phase(phase)
        s = enc(ev.x)
        if ev.mode == "sup":
            out = agent.sup(s, ev.y)
        elif ev.mode == "reward":
            out = agent.reward(s, ev.y)
        else:
            out = agent.unlab(s, ev.y)
        hits[ev.t] = out == ev.y; ys[ev.t] = ev.y; newf[ev.t] = ev.new
        own = SCHEDULE[ev.phase].classes                   # acierto restringido a las clases de la fase
        z = np.asarray(agent.last_scores())
        hits_r[ev.t] = own[int(np.argmax(z[list(own)]))] == ev.y
        if ev.phase in keep_from and ev.t - bounds[ev.phase][0] < W:
            saved[keep_from[ev.phase]].append((s, ev.y))
        if ev.phase == probe_k:
            saved["P"].append((s, ev.y))
        if (ev.t + 1) % log_every == 0:
            growth.append(dict(t=ev.t + 1, **agent.stats()))
        if ev.t + 1 == bounds[ev.phase][1]:
            acc = evaluate(agent, evals)
            checkpoints.append(dict(phase=SCHEDULE[ev.phase].name, t=ev.t + 1, acc=acc, **agent.stats()))
            print(f"[{name} s{seed}] {SCHEDULE[ev.phase].name:18s} t={ev.t + 1:7d} "
                  f"clean={acc['clean/known']:.3f} rot={acc['rot90/known']:.3f} "
                  f"occl={acc['occlude/known']:.3f} P={acc['clean/P']:.3f} "
                  f"preq={hits[bounds[ev.phase][0]:ev.t + 1].mean():.3f} "
                  f"n={agent.stats()['neurons']} {time.time() - t0:.0f}s", flush=True)
    train_s = time.time() - t0

    # ---- redes nuevas sobre los mismos eventos (transferencia y plasticidad al principio)
    fresh = {}
    for g, evs in saved.items():
        a = factory(seed)
        a.on_phase(0)
        own = list(SCHEDULE[probe_k].classes if g == "P" else GROUPS[g])
        h, hr = [], []
        for s, y in evs:
            h.append(a.sup(s, y) == y)
            hr.append(own[int(np.argmax(np.asarray(a.last_scores())[own]))] == y)
        rec = dict(hits=np.array(h, dtype=np.int8).tolist(), hits_r=np.array(hr, dtype=np.int8).tolist(),
                   labels=[int(y) for _, y in evs])
        if g == "P":
            rec["acc"] = evaluate(a, {"clean": evals["clean"]})
        fresh[g] = rec

    return dict(model=name, seed=seed, scale=scale, T=T, W=W, n_eval=n_eval,
                bounds=bounds, phases=[p.name for p in SCHEDULE], train_seconds=train_s,
                hits=hits.tolist(), hits_r=hits_r.tolist(), labels=ys.tolist(), new=newf.tolist(),
                growth=growth, checkpoints=checkpoints, fresh=fresh)


# ====================================================================== métricas
def metrics(r: dict) -> dict:
    hits = np.array(r["hits"]); ys = np.array(r["labels"]); new = np.array(r["new"], dtype=bool)
    bounds = r["bounds"]; W = r["W"]
    cps = r["checkpoints"]
    m = {"model": r["model"], "seed": r["seed"], "scale": r["scale"], "T": r["T"]}
    final = cps[-1]
    pre_probe = cps[-2]
    m["final_clean_known"] = final["acc"]["clean/known"]
    m["preprobe_clean_known"] = pre_probe["acc"]["clean/known"]
    m["final_rot90"] = final["acc"]["rot90/known"]
    m["final_occlude"] = final["acc"]["occlude/known"]
    m["final_noise"] = final["acc"]["noise/known"]
    # olvido: sólo puntos donde el grupo ya se había entrenado en ese contexto
    hist = {}
    for ctx in ("clean", "rot90", "occlude"):
        hist[ctx] = []
        for cp in cps[:-1]:                                   # antes del sondeo
            hist[ctx].append({g: cp["acc"][f"{ctx}/{g}"] for g in EVAL_SETS[ctx] if g != "P"})
    for ctx in ("clean", "rot90", "occlude"):
        f = forgetting(hist[ctx])
        m[f"forget_{ctx}"] = float(np.mean(list(f.values()))) if f else float("nan")
    f_all = forgetting(hist["clean"] + [{g: final["acc"][f"clean/{g}"] for g in "ABCD"}])
    m["forget_clean_with_probe"] = float(np.mean(list(f_all.values())))
    # ahorro al reaparecer (mismo modo y contexto)
    sav = []
    for k0, k1, ctx in reappearances():
        a0 = window_acc(hits, bounds[k0][0], W); a1 = window_acc(hits, bounds[k1][0], W)
        name = r["phases"][k1]
        m[f"first_{name}"] = a0; m[f"back_{name}"] = a1
        s0 = steps_to(hits, bounds[k0][0], bounds[k0][1], 0.9, W // 2)
        s1 = steps_to(hits, bounds[k1][0], bounds[k1][1], 0.9, W // 2)
        m[f"steps90_first_{name}"] = s0; m[f"steps90_back_{name}"] = s1
        sav.append(a1 - a0)
    m["savings"] = float(np.mean(sav))
    # transferencia hacia adelante: primeros W pasos de B, C, D vs red nueva
    fw = []
    first = first_appearance()
    hits_r = np.array(r["hits_r"])
    fwr = []
    for g in ("B", "C", "D"):
        k = first[g]
        a_stream = window_acc(hits, bounds[k][0], W)
        a_fresh = float(np.mean(r["fresh"][g]["hits"]))
        m[f"fwt_{g}"] = a_stream - a_fresh
        fw.append(a_stream - a_fresh)
        fwr.append(window_acc(hits_r, bounds[k][0], W) - float(np.mean(r["fresh"][g]["hits_r"])))
    m["fwt"] = float(np.mean(fw))
    m["fwt_restricted"] = float(np.mean(fwr))
    # plasticidad tardía: sondeo P en la red entrenada vs red nueva
    k = len(r["phases"]) - 1
    seg = slice(bounds[k][0], bounds[k][1])
    pn = new[seg]
    m["probe_preq_trained"] = float(hits[seg][pn].mean())
    fh = np.array(r["fresh"]["P"]["hits"]); fl = np.array(r["fresh"]["P"]["labels"])
    fnew = np.isin(fl, GROUPS["P"])
    m["probe_preq_fresh"] = float(fh[fnew].mean())
    n_first = min(int(pn.sum()), W)
    m["probe_first_trained"] = float(hits[seg][pn][:n_first].mean())
    m["probe_first_fresh"] = float(fh[fnew][:n_first].mean())
    fhr = np.array(r["fresh"]["P"]["hits_r"])
    m["probe_preq_restricted_trained"] = float(hits_r[seg][pn].mean())
    m["probe_preq_restricted_fresh"] = float(fhr[fnew].mean())
    m["probe_test_trained"] = final["acc"]["clean/P"]
    m["probe_test_fresh"] = r["fresh"]["P"]["acc"]["clean/P"]
    m["probe_restricted_trained"] = final["acc"]["clean/P_restricted"]
    m["probe_restricted_fresh"] = r["fresh"]["P"]["acc"]["clean/P_restricted"]
    m["plasticity_ratio"] = m["probe_test_trained"] / max(m["probe_test_fresh"], 1e-9)
    # fases sin etiquetas y sólo con recompensa
    names = r["phases"]
    ku = names.index("BC_unlab_noise")
    for g in ("B", "C"):
        m[f"unlab_drift_{g}"] = cps[ku]["acc"][f"clean/{g}"] - cps[ku - 1]["acc"][f"clean/{g}"]
    for name in ("A_reward", "all_reward_noise"):
        kk = names.index(name)
        m[f"preq_{name}"] = float(hits[bounds[kk][0]:bounds[kk][1]].mean())
    # crecimiento
    m["neurons_end"] = final["neurons"]; m["features_end"] = final["features"]
    m["bytes_end"] = final["bytes"]
    m["neurons_mid"] = cps[len(cps) // 2]["neurons"]
    m["train_seconds"] = r["train_seconds"]
    return m


def summarize(pattern: str = "*.json"):
    rows = [metrics(json.loads(p.read_text())) for p in sorted(OUT.glob(pattern))]
    if not rows:
        return
    keys = list(rows[0].keys())
    log = CsvLog("continual_summary", keys)
    for r in rows:
        log.row(*[r.get(k, "") for k in keys])
    log.close()
    # agregado por (modelo, escala): media y desvío entre semillas
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["scale"]), []).append(r)
    num = [k for k in keys if k not in ("model", "seed", "scale") and isinstance(rows[0][k], (int, float))]
    agg = CsvLog("continual_agg", ["model", "scale", "seeds"] + [f"{k}" for k in num])
    for (model, scale), rs in sorted(groups.items()):
        vals = []
        for k in num:
            v = np.array([x[k] for x in rs], dtype=float)
            vals.append(f"{np.nanmean(v):.4f}±{np.nanstd(v):.4f}")
        agg.row(model, scale, len(rs), *vals)
    agg.close()


def load_data():
    Xm, ym, Xmt, ymt = load_mnist()
    Xf, yf, Xft, yft = load_fashion()
    pools = {c: Xm[ym == c] for c in range(10)}
    pools.update({10 + c: Xf[yf == c] for c in range(10)})
    test = {c: Xmt[ymt == c] for c in range(10)}
    test.update({10 + c: Xft[yft == c] for c in range(10)})
    return pools, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["d1", "d2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--pattern", default="*.json")
    a = ap.parse_args()
    if a.summary:
        summarize(a.pattern)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    for name in a.models:
        for seed in a.seeds:
            r = run(name, seed, a.scale, a.n_eval, data)
            path = OUT / f"{name}_s{seed}_x{a.scale:g}.json"
            path.write_text(json.dumps(r))
            m = metrics(r)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}),
                  flush=True)


if __name__ == "__main__":
    main()
