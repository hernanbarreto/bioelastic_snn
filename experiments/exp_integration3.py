"""Integración, ronda 3: dónde y por qué falla L3 (percepción + razonamiento desde cero).

L3: dos números de 1 a 4 dígitos (imágenes MNIST) se suman columna por columna; la
percepción (Cortex depth2, desde cero) convierte cada imagen en un símbolo, el Thinker
(desde cero, 2 registros) responde el dígito de la columna y, al final, el último acarreo.
La única señal es el dígito correcto de cada columna. Mejor variante de la ronda 2
(cred2+H): exacta n=3 0.000-0.040, columna 0.17-0.37.

Fases (cada una escribe results/integracion3_<fase>_<variante>_seed<S>.csv en formato
largo y `resumen` junta todo en results/integracion3_<fase>.json con media y desvío):

  f0   localizar el cuello de botella: reproducción de cred2+H con descomposición
       (Thinker con el código final de la percepción como oráculo), controles
       (a) percepción oráculo / con ruido estacionario / congelada en el código final, con
       Thinker que aprende; (b) Thinker oráculo (suma correcta fija) con percepción que
       aprende sólo del dígito final; (c) objetivos barajados; (d) azar e información
       mutua analíticos (`analisis`).
  f1   crédito posterior y aprendizaje trans-situacional (ver `CreditoPosterior`).
  f2   metacognición: calibración (ECE), detección del error propio antes del feedback
       (AUROC) y plasticidad ponderada por confianza (con y sin).
  f3   incertidumbre esperada (ACh) e inesperada (NE) con un cambio de regla a mitad.
  f4   currículo sólo por los datos presentados vs. orden aleatorio, mismo presupuesto.
  f5   pruebas adversariales: combinaciones nunca vistas, símbolo nuevo tardío,
       latencia y costo, aprendizaje que no se detiene (test-then-train).

  python experiments/exp_integration3.py <fase> [--seed S] [--variante V] [--rapida]
  python experiments/exp_integration3.py resumen

Toda evaluación usa ejemplares del split oficial de test (MNIST t10k); lo que se usa para
elegir umbrales sale de train. Ninguna cifra de acá está validada: es lo medido.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from itertools import product

import numpy as np

import exp_integration as ei
import exp_integration2 as e2
from common import RESULTS, CsvLog
from bioelastic.thinker import Thinker

HEADER = ei.HEADER
END = ei.END


# ====================================================================== análisis
def p_acarreo(n: int) -> float:
    """P(el acarreo que sale de la columna n es 1) con dígitos uniformes e independientes."""
    p = 0.0
    for _ in range(n):
        p = 0.45 + 0.10 * p                        # P(a+b+c >= 10) = (45 + 10c) / 100
    return p


def _mi(joint: np.ndarray) -> float:
    """Información mutua (bits) de una tabla conjunta P[x, y]."""
    joint = joint / joint.sum()
    px = joint.sum(1, keepdims=True); py = joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (px @ py)[nz])).sum())


def analisis() -> dict:
    """Azar y cuánta información lleva cada ranura sobre la respuesta, por enumeración.

    La suma módulo 10 es un cuadrado latino: con el otro sumando uniforme, el dígito de
    la columna es independiente de cada sumando por separado (I(a; d) = 0) aunque los dos
    juntos lo determinan. Cualquier regla que reparta el crédito por ranura (recompensa
    por imagen, explicación de una sola ranura, co-ocurrencia imagen-respuesta) no tiene
    señal media mientras la otra ranura esté al azar. El acarreo sí depende de cada sumando
    por separado (magnitud)."""
    A = np.arange(10)
    jd = np.zeros((10, 10)); jab = np.zeros((100, 10)); jdb = np.zeros((100, 10)); jc = np.zeros((10, 2))
    for a, b in product(A, A):
        d = (a + b) % 10
        jd[a, d] += 1; jab[a * 10 + b, d] += 1; jdb[b * 10 + a, d] += 1
        jc[a, int(a + b >= 10)] += 1
    out = dict(
        azar_columna=0.1,
        p_acarreo={n: p_acarreo(n) for n in (1, 2, 3, 4)},
        azar_exacta_n3_mejor_constante=0.1 ** 3 * max(p_acarreo(3), 1 - p_acarreo(3)),
        azar_exacta_n3_uniforme=0.1 ** 4,
        bits_I_a_d=_mi(jd),                       # L2/L3: una ranura sola
        bits_I_ab_d=_mi(jab),                     # las dos juntas
        bits_I_a_d_dado_b=_mi(jab) - _mi(jdb.reshape(10, 10, 10).sum(1)),   # L1: I(a; d | b) = I(a,b; d) - I(b; d)
        bits_I_a_acarreo=_mi(jc),                 # el acarreo sí dice algo de una ranura
        nota=("I(a;d)=0: sin señal por ranura al azar; con la otra ranura con exactitud q "
              "la ventaja de acertar una ranura es (10q-1)/9 (0 en q=0.1). EM exacto con "
              "conocimiento correcto: q -> q^2/(q^2+(1-q)^2/9), punto fijo inestable en 0.1 "
              "(derivada 2)."),
    )
    return out


def ventaja_por_ranura(q: float) -> float:
    """P(acierto | ranura a bien) - P(acierto | ranura a mal) con la otra ranura con
    exactitud q (errores uniformes) y razonamiento correcto."""
    return q - (1 - q) / 9


# ====================================================================== percepciones
class Percepcion3(e2.Percepcion):
    """La percepción de la ronda 2 (símbolo, certeza, k mejores, `ensenar`), que además
    deja a mano la distribución completa sobre los símbolos (`p`) y qué eligió cada
    votante del Cortex (`votos`: área de asociación, proyección, lector local)."""

    def ver(self, img, d, explore):
        s, c, alts = super().ver(img, d, explore)
        net = self.v.net
        self.p = net._p().astype(np.float64)
        self.votos = net._L.argmax(1) if net._L is not None else np.array([s])
        self.L = None if net._L is None else net._L.astype(np.float64)
        self.conflicto = net.conflict()
        return s, c, alts


def _ensenar_suave(self, img, q):
    """La percepción aprende de la imagen con un objetivo blando (posterior): q - p como
    tercer factor en cada regla local (`Cortex.teach(soft=q)`)."""
    self.v.net.forward(self.v._E(img))
    self.v.net.teach(soft=np.asarray(q, dtype=np.float64))
    self.ensenadas += 1


Percepcion3.ensenar_suave = _ensenar_suave


class OraculoP:
    """Percepción perfecta: el símbolo es el dígito (o `pi[dígito]`: el código de otra
    percepción usado como oráculo)."""
    k = 1

    def __init__(self, pi=None):
        self.pi = None if pi is None else np.asarray(pi)

    def ver(self, img, d, explore):
        s = int(d) if self.pi is None else int(self.pi[int(d)])
        self.p = np.eye(10)[s]
        self.votos = np.array([s])
        return s, 1.0, []


class RuidosaP:
    """Símbolos con exactitud fija `acc` (los errores, uniformes sobre los otros 9): ruido
    estacionario, sin imágenes."""
    k = 1

    def __init__(self, acc: float, seed: int):
        self.acc = float(acc)
        self.rng = np.random.default_rng(seed)

    def ver(self, img, d, explore):
        d = int(d)
        s = d if self.rng.random() < self.acc else int((d + 1 + self.rng.integers(9)) % 10)
        self.p = np.full(10, (1 - self.acc) / 9); self.p[s] = self.acc
        self.votos = np.array([s])
        return s, self.acc, []


class Congelada(Percepcion3):
    """Una percepción que ya no aprende: decide por argmax y no recibe crédito."""

    def ver(self, img, d, explore):
        return super().ver(img, d, False)

    def ensenar(self, img, sym):
        pass

    def reforzar(self, img, sym, r):
        pass


# ====================================================================== Thinker oráculo
class ThinkerOraculo:
    """Conocimiento correcto y fijo: suma decimal sobre los símbolos que recibe (el
    acarreo sale de lo percibido, así que un error de percepción lo arrastra). No aprende.
    Con `alternatives` explica el error como el Thinker de la ronda 2 con `explain=
    "refutadas"` (todo su conocimiento está establecido): si una sola ranura (con sus
    alternativas) o el acarreo lo explican, `input_credit` dice qué símbolo debió verse."""

    def __init__(self):
        self.plastic = True
        self.abstained = 0; self.blamed_inputs = 0
        self.input_credit = []
        self.reset()

    def reset(self):
        self.c = 0
        self._q = None

    def answer(self, tokens, cue=0, explore=False, certainty=1.0, probe=None):
        if tokens == END:
            self._q = ("fin", self.c)
            self._out = self.c
            return self.c
        sa, sb = int(tokens[0]), int(tokens[1]) - 10
        self._q = ((sa, sb), self.c)
        s = sa + sb + self.c
        self.c = s // 10
        self._out = s % 10
        return self._out

    def state_key(self):
        return (int(self._q[1]),)

    def answer_distribution(self, pa, pb) -> np.ndarray:
        """P(dígito) con la suma correcta, la percepción incierta y su acarreo actual."""
        J = np.asarray(pa)[:, None] * np.asarray(pb)[None, :]
        A = np.arange(10)
        return np.bincount(((A[:, None] + A[None, :] + self.c) % 10).ravel(), weights=J.ravel(), minlength=10)

    def feedback(self, target=None, reward=None, alternatives=None, trust=None):
        self.input_credit = []
        if alternatives is None or self._q is None or self._q[0] == "fin":
            return
        (sa, sb), c = self._q
        if (sa + sb + c) % 10 == target:
            return
        expl = []
        for i, alts in enumerate(alternatives):
            for s in alts:
                x = [sa, sb]; x[i] = int(s) - 10 * i
                if (x[0] + x[1] + c) % 10 == target:
                    expl.append((i, int(s)))
        if (sa + sb + 1 - c) % 10 == target:
            expl.append((-1, 1 - c))
        if len(expl) == 1 and expl[0][0] >= 0:
            self.input_credit = [expl[0]]
            self.blamed_inputs += 1

    def consistency(self, target, candidates, regs=None):
        """Como `Thinker.consistency`: +1 si la suma da el objetivo, -1 si no (todo su
        conocimiento está establecido). `regs`: acarreos a considerar (por defecto el que
        tenía al responder)."""
        _, c = self._q
        regs = [c] if regs is None else [int(np.asarray(r).ravel()[0]) for r in regs]
        ca = np.asarray(candidates[0]); cb = np.asarray(candidates[1]) - 10
        return np.stack([np.where((ca[:, None] + cb[None, :] + r) % 10 == int(target), 1, -1)
                         for r in regs]).astype(np.int8)

    def stats(self) -> dict:
        return dict(abstained=self.abstained, blamed_inputs=self.blamed_inputs)


# ====================================================================== F1: crédito posterior
class CreditoPosterior:
    """Adaptador de `bioelastic.posterior.PosteriorCredit` al episodio L3: le pasa al
    crédito posterior lo percibido y la distribución de la percepción en cada ranura, y
    le enseña a la percepción lo que el posterior decida. `refuerzo`: además, +-1 por
    columna a la percepción (como la ronda 2). `peso(sym, percibido, q)` (F2): cuántas
    veces repetir cada enseñanza (plasticidad ponderada)."""

    def __init__(self, refuerzo=False, fin=False, **kw):
        from bioelastic.posterior import PosteriorCredit
        self.pc = PosteriorCredit(**kw)
        self.refuerzo = bool(refuerzo)
        self.fin_ = bool(fin)              # sólo con el Thinker oráculo: la última columna también
        self.ultima = None                 # se explica con el acarreo final observado
        self.peso = None

    @property
    def n(self):
        return self.pc.n

    def empezar(self):
        pass

    def columna(self, th, per, imgs, syms, priors, y, alts, st, digitos):
        out = th._out
        if self.fin_ and isinstance(th, ThinkerOraculo):
            self.ultima = (imgs, syms, priors, y, th._q[1], digitos)
        res = self.pc.step(th, syms, priors, y)
        for img, t, s, dg in zip(imgs, res["teach"], syms, digitos):
            if t is None:
                continue
            sym, w = t
            reps = 1 if self.peso is None else int(self.peso(sym, s, w))
            for _ in range(reps):
                per.ensenar(img, sym)
            if reps:
                st["creditos"] = st.get("creditos", 0) + 1
                st["creditos_bien"] = st.get("creditos_bien", 0) + int(sym == int(dg))
                st["auto"] = st.get("auto", 0) + int(sym == s)
                st.setdefault("pares", []).append((int(dg), sym))
        if self.refuerzo:
            r = 1.0 if out == y else -1.0
            for img, s in zip(imgs, syms):
                per.reforzar(img, s, r)

    def fin(self, th, y, st=None):
        th.feedback(target=y)
        if not (self.fin_ and self.ultima is not None):
            return
        from bioelastic.posterior import joint_posterior
        imgs, syms, priors, d, c, digitos = self.ultima
        self.ultima = None
        A = np.arange(10)
        L = (((A[:, None] + A[None, :] + c) % 10 == d) & ((A[:, None] + A[None, :] + c >= 10) == bool(y))).astype(float)
        q, margs = joint_posterior(list(priors), L, self.pc.gamma)
        for img, qm, s, dg in zip(imgs, margs, syms, digitos):
            j = int(self.pc.rng.choice(10, p=qm / qm.sum())) if self.pc.mode == "sample" else int(np.argmax(qm))
            if self.pc.mode == "map" and qm[j] < self.pc.tau:
                continue
            per_ = self._per
            per_.ensenar(img, j)
            if st is not None:
                st["creditos"] = st.get("creditos", 0) + 1
                st["creditos_bien"] = st.get("creditos_bien", 0) + int(j == int(dg))
                st.setdefault("pares", []).append((int(dg), j))


class CreditoCadena:
    """F1 con posterior CONJUNTO: al terminar el episodio, el razonamiento calcula el
    posterior sobre todas las asignaciones (cada dígito de cada columna y los acarreos)
    dados todos los resultados observados, incluido el acarreo final
    (`bioelastic.posterior.chain_posterior`, inferencia exacta hacia adelante y hacia
    atrás por la cadena de acarreos), combinando con p(símbolo | imagen). La percepción
    aprende cada imagen con su marginal: objetivo blando (q - p como tercer factor) o una
    muestra de q (EM estocástico). regla="conocida": la suma (como DeepProbLog);
    "aprendida": la del razonador `ChainRule`, que se aprende a la vez (abducción blanda,
    como Abductive Learning)."""

    def __init__(self, regla="conocida", modo="blando", gamma=1.0, compuerta_ne=None, equiparticion=0.0,
                 fiabilidad=None, seed=0):
        self.regla, self.modo, self.gamma = regla, modo, float(gamma)
        # fiabilidad (sólo con la regla conocida, donde los símbolos tienen significado
        # fijo): gamma = r^k con r = acierto por columna del razonamiento sobre lo percibido,
        # por encima del azar; dict(k, rate, lo). Ver exp_mnist_addition.Fiabilidad.
        self.fiab = None
        if fiabilidad is not None:
            from bioelastic.posterior import ReliabilityGate
            self.fiab = ReliabilityGate(**dict(dict(chance=0.1, k=2.0, rate=0.01, lo=0.02), **fiabilidad))
            self.gamma = self.fiab.gamma
        from bioelastic.posterior import Equipartition
        # prior de equipartición sobre los símbolos (contra fusiones de conceptos)
        self.eq = Equipartition(10, equiparticion) if equiparticion > 0 else None
        # con NE del razonador por encima de `compuerta_ne`, la percepción no aprende de este
        # episodio: la sorpresa inesperada se atribuye a un cambio de la regla, no a lo visto
        self.compuerta_ne = compuerta_ne
        self.rng = np.random.default_rng(20_000 + seed)
        self.n = dict(episodios=0, ensenadas=0, verosimilitud_media=0.0)
        self.cols = []

    def empezar(self):
        self.cols = []

    def columna(self, th, per, imgs, syms, priors, y, alts, st, digitos):
        if self.fiab is not None:
            self.fiab.update(th._out == y)
        th.feedback(target=y)
        self.cols.append((imgs, priors, int(y), digitos, syms))

    def fin(self, th, y, st=None):
        from bioelastic.posterior import chain_posterior, known_sum_factor
        th.feedback(target=y)
        if not self.cols:
            return
        pa = [c[1][0] for c in self.cols]; pb = [c[1][1] for c in self.cols]
        if self.eq is not None:
            pa = [self.eq(x) for x in pa]; pb = [self.eq(x) for x in pb]
        digs = [c[2] for c in self.cols]
        if self.regla == "conocida":
            qa, qb, _, _, lik = chain_posterior(pa, pb, [known_sum_factor(d) for d in digs], np.eye(2)[int(y)],
                                                self.gamma)
        else:
            qa, qb, _, _, lik = th.learn(pa, pb, digs, int(y), self.gamma)
        self.n["episodios"] += 1
        self.n["verosimilitud_media"] += (lik - self.n["verosimilitud_media"]) / self.n["episodios"]
        if self.fiab is not None:
            self.gamma = self.fiab.gamma
            self.n["gamma"] = self.gamma
        per = self._per
        if self.compuerta_ne is not None and getattr(th, "unexpected_uncertainty", 0.0) > self.compuerta_ne:
            self.n["compuerta"] = self.n.get("compuerta", 0) + 1
            self.cols = []
            return
        for (imgs, _, _, digitos, syms), xa, xb in zip(self.cols, qa, qb):
            for img, q, dg in zip(imgs, (xa, xb), digitos):
                if self.modo == "muestreo":
                    j = int(self.rng.choice(10, p=q / q.sum()))
                    per.ensenar(img, j)
                else:
                    j = int(np.argmax(q))
                    per.ensenar_suave(img, q)
                self.n["ensenadas"] += 1
                if st is not None:
                    st["creditos"] = st.get("creditos", 0) + 1
                    st["creditos_bien"] = st.get("creditos_bien", 0) + int(j == int(dg))
                    st.setdefault("pares", []).append((int(dg), j))
        self.cols = []


# ====================================================================== episodio L3
def episodio(th, per, datos, split, rng, n, learn, cfg, st, objetivo=None, registro=None, digitos=None):
    """Suma de dos números de n dígitos columna por columna (mismo orden de sorteos que
    `exp_integration2.episodio_suma`: con cfg de la ronda 2 reproduce sus corridas bit a
    bit). cfg: cred (alternativas + crédito simbólico), refuerzo (+-1 por columna a la
    percepción), conf (certeza a la pregunta), posterior (objeto `CreditoPosterior`).
    `objetivo(i, n, d)`: qué objetivo recibe la red en lugar del verdadero (control de
    etiquetas barajadas; la evaluación siempre usa el verdadero). `registro`: lista donde
    se agrega un dict por columna (para calibración y detección de errores). `digitos`:
    (a, b) dados (currículo, pruebas adversariales) en vez de sorteados."""
    if digitos is None:
        a = rng.integers(10, size=n); b = rng.integers(10, size=n)
    else:
        a, b = (np.asarray(x, dtype=np.int64) for x in digitos); n = len(a)
    regla = cfg.get("regla")
    th.reset()
    carry = 0; cols = []; vistos = []
    post = cfg.get("posterior")
    if post is not None:
        post.empezar()
    for i in range(n + 1):
        alts = None; info = None
        if i < n:
            ia, ib = datos.una(split, a[i], rng), datos.una(split, b[i], rng)
            sa, ca, aa = per.ver(ia, a[i], learn)
            pa = getattr(per, "p", None); va = getattr(per, "votos", None)
            per.conflicto_a = getattr(per, "conflicto", None)
            sb, cb, ab = per.ver(ib, b[i], learn)
            pb = getattr(per, "p", None); vb = getattr(per, "votos", None)
            per.conflicto_b = getattr(per, "conflicto", None)
            vistos += [sa == a[i], sb == b[i]]
            s = int(a[i]) + int(b[i]) + carry
            d, carry = s % 10, s // 10
            if regla:                                     # F3: otra regla para el dígito (el acarreo no cambia)
                d = (d + regla) % 10
            cert = min(ca, cb) if cfg.get("conf") else 1.0
            pdist = th.answer_distribution(pa, pb) if (registro is not None and pa is not None
                                                       and hasattr(th, "answer_distribution")) else None
            out = th.answer((sa, 10 + sb), explore=learn, certainty=cert)
            if cfg.get("cred"):
                alts = [aa, e2._desplazar(ab, 10)]
            info = dict(sa=sa, sb=sb, pa=pa, pb=pb, va=va, vb=vb, a=int(a[i]), b=int(b[i]),
                        ca=getattr(per, "conflicto_a", None), cb=getattr(per, "conflicto_b", None),
                        p_sis=None if pdist is None else float(pdist[out]))
        else:
            d = carry
            out = th.answer(END, explore=learn)
        if registro is not None:
            registro.append(dict(col=i, fin=i == n, out=out, d=d, ok=out == d,
                                 **(_senales(th, per, info) if info else {})))
        y = d if objetivo is None else objetivo(i, n, d)
        if learn:
            if post is not None and i < n:
                post.columna(th, per, (ia, ib), (sa, sb), (pa, pb), y, alts, st, (a[i], b[i]))
            else:
                th.feedback(target=y, alternatives=alts)
                if i < n:
                    e2._credito(th, per, (ia, ib), (0, 10), (a[i], b[i]), st)
                    if cfg.get("refuerzo"):
                        r = 1.0 if out == y else -1.0
                        per.reforzar(ia, sa, r); per.reforzar(ib, sb, r)
            if post is not None and i == n:
                post._per = per
                post.fin(th, y, st)
        cols.append(out == d)
    return dict(exacta=all(cols), columnas=cols, vistos=vistos)


def _senales(th, per, info) -> dict:
    """Señales internas antes del feedback (F2). Percepción: probabilidad de lo elegido,
    conflicto entre los votantes del Cortex (`Cortex.conflict`) y acuerdo de sus argmax.
    Thinker: probabilidad de su respuesta en la lectura del conocimiento y si es
    conocimiento establecido. Entre las dos: conflicto predictivo (probabilidad, según la
    percepción, de que lo visto lleve con conocimiento establecido a otra respuesta)."""
    out = dict(dig_a=info["a"], dig_b=info["b"], s_a=info["sa"], s_b=info["sb"])
    for k in ("a", "b"):
        p = info[f"p{k}"]; s = info[f"s{k}"]; v = info[f"v{k}"]
        if p is not None:
            out[f"pp{k}"] = float(p[s])
        if v is not None and len(v) > 1:
            out[f"acuerdo{k}"] = float(np.mean(v == s))
        c = info.get(f"c{k}")
        if c is not None:
            out[f"conf_cx{k}"] = float(c)
    if info.get("p_sis") is not None:                     # confianza del sistema: P(respuesta | percepción, regla)
        out["p_sis"] = info["p_sis"]
    if isinstance(th, Thinker):
        c, sure = th.answer_confidence()
        out["pt"] = float(c); out["tsure"] = float(sure)
        if info["pa"] is not None:
            from bioelastic.posterior import predictive_conflict, top_candidates
            ca = top_candidates(info["pa"], 3, keep=info["sa"]); cb = top_candidates(info["pb"], 3, keep=info["sb"])
            outs, sure_ = th.content_outputs([ca, cb + 10])
            out["conflicto_pt"] = predictive_conflict(outs, sure_, [info["pa"][ca], info["pb"][cb]], th._out)
    return out


def entrenar(th, per, datos, rng, episodios, cfg, st, objetivo=None, curva=None, cada=500,
             Xc=None, yc=None, n_rango=(1, 5), registro=None, generador=None, al_episodio=None):
    """Episodios de 1 a 4 dígitos (o los que dé `generador(e, rng)`: n, o (a, b) dígitos).
    `al_episodio(e, resultado)` se llama después de cada episodio."""
    win = []
    for e in range(episodios):
        g = None if generador is None else generador(e, rng)
        if g is None or isinstance(g, (int, np.integer)):
            n = int(rng.integers(*n_rango)) if g is None else int(g)
            r = episodio(th, per, datos, "train", rng, n, True, cfg, st, objetivo, registro)
        else:
            r = episodio(th, per, datos, "train", rng, 0, True, cfg, st, objetivo, registro, digitos=g)
        if al_episodio is not None:
            al_episodio(e, r)
        win += r["columnas"]
        if curva is not None and (e + 1) % cada == 0:
            fila = dict(ep=e + 1, columna=float(np.mean(win)))
            if Xc is not None and hasattr(per, "v"):
                conf = per.v.confusion(Xc, yc)
                fila.update(pureza=ei.pureza(conf), identidad=float(np.trace(conf) / conf.sum()),
                            usados=int((conf.sum(0) > 0).sum()))
            curva.append(fila); win = []


def codigo(conf: np.ndarray) -> np.ndarray:
    """Mejor biyección dígito -> símbolo de una matriz de confusión conf[dígito, símbolo]."""
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-conf)
    pi = np.zeros(10, dtype=np.int64); pi[r] = c
    return pi


SPLIT_EVAL = ["test"]           # "pre" en las corridas de validación (--val): train[:n_pre], que L3 no usa


def evaluar(th, per, datos, rng, n, episodios, pi=None) -> dict:
    """Evaluación sin aprender (test, o validación con --val)."""
    ei.plasticidad(th, False)
    res = [episodio(th, per, datos, SPLIT_EVAL[0], rng, n, False, {}, {}) for _ in range(episodios)]
    ei.plasticidad(th, True)
    ex = np.array([r["exacta"] for r in res])
    return dict(exacta=float(ex.mean()),
                columna=float(np.mean([c for r in res for c in r["columnas"][:-1]])),
                fin=float(np.mean([r["columnas"][-1] for r in res])))


def evaluar_con_codigo(th, per, datos, rng, n, episodios, pi) -> dict:
    """Como `evaluar`, pero además separa los episodios con todas las imágenes vistas
    según el código `pi` (exacta | todas bien) de los demás."""
    ei.plasticidad(th, False)
    ex, ok = [], []
    for _ in range(episodios):
        a = rng.integers(10, size=n); b = rng.integers(10, size=n)
        th.reset(); carry = 0; cols = []; bien = True
        for i in range(n + 1):
            if i < n:
                ia, ib = datos.una(SPLIT_EVAL[0], a[i], rng), datos.una(SPLIT_EVAL[0], b[i], rng)
                sa = per.ver(ia, a[i], False)[0]; sb = per.ver(ib, b[i], False)[0]
                bien &= (sa == pi[a[i]]) and (sb == pi[b[i]])
                s = int(a[i]) + int(b[i]) + carry; d, carry = s % 10, s // 10
                out = th.answer((sa, 10 + sb))
            else:
                d = carry; out = th.answer(END)
            cols.append(out == d)
        ex.append(all(cols)); ok.append(bien)
    ei.plasticidad(th, True)
    ex = np.array(ex); ok = np.array(ok)
    return dict(todas_bien=float(ok.mean()),
                exacta_si_todas_bien=float(ex[ok].mean()) if ok.any() else float("nan"),
                exacta_si_alguna_mal=float(ex[~ok].mean()) if (~ok).any() else float("nan"))


def thinker_l3(seed, **kw):
    """El Thinker de L3 de la ronda 2 (21 símbolos, 2 registros que empiezan en 0)."""
    return Thinker(21, registers=2, seed=seed, **kw)


def medir_percepcion(out, fase, var, v, Xt, yt, extra=None) -> dict:
    conf = v.confusion(Xt, yt)
    pz = ei.pureza(conf); ident = float(np.trace(conf) / conf.sum())
    pi = codigo(conf)
    out(fase, var, "pureza percepción (test)", pz, 0.1, "mejor biyección")
    out(fase, var, "identidad percepción (test)", ident, 0.1, "símbolo == dígito")
    out(fase, var, "símbolos usados", int((conf.sum(0) > 0).sum()), 10, "")
    out(fase, var, "código es identidad (dígitos)", int((pi == np.arange(10)).sum()), 10,
        "dígitos cuyo símbolo en la mejor biyección es el propio dígito")
    if extra is not None:
        extra[var] = dict(confusion=conf.tolist(), codigo=pi.tolist())
    return dict(pureza=pz, identidad=ident, pi=pi, conf=conf)


def objetivos_barajados(seed):
    """Control (c): objetivo al azar con la misma marginal que el verdadero."""
    r = np.random.default_rng(777 + seed)

    def f(i, n, d):
        if i < n:
            return int(r.integers(10))
        return int(r.random() < p_acarreo(n))
    return f


# ====================================================================== F0
def fase_f0(seed, tam, out, variante, extra):
    """Localizar el cuello de botella (ver docstring del módulo)."""
    datos = ei.Digitos(tam["n_pre"])
    rng0 = np.random.default_rng(2000 + seed)
    Xt, yt = ei.muestra_test(datos, rng0, tam["n_test_pureza"])
    Xc, yc = Xt[:tam["n_curva"]], yt[:tam["n_curva"]]
    variantes = tam["f0"] if variante is None else [variante]
    for var in variantes:
        rng = np.random.default_rng(2000 + seed)
        ei.muestra_test(datos, rng, tam["n_test_pureza"])       # mismo estado del rng que la ronda 2
        t0 = time.time(); st: dict = {}; curva: list = []
        if var in ("rep", "rep_v02", "c_barajadas"):
            # rep_v02: la percepción con la que se midió la ronda 2 (V1 sin error local ni
            # árbitro: el depth2 de antes de integrar la rama de profundidad)
            v = ei.Vista(seed, config="depth2_v02" if var == "rep_v02" else "depth2",
                         homeostasis=tam["homeostasis"])
            per = Percepcion3(v, k=2)
            th = thinker_l3(seed)
            obj = objetivos_barajados(seed) if var == "c_barajadas" else None
            entrenar(th, per, datos, rng, tam["ep"], dict(cred=True, refuerzo=True), st, obj, curva,
                     tam["cada"], Xc, yc)
        elif var.startswith("b_"):
            v = ei.Vista(seed, homeostasis=tam["homeostasis"])
            per = Percepcion3(v, k=2)
            th = ThinkerOraculo()
            cfg = dict(refuerzo=True, cred=var == "b_cred2")
            entrenar(th, per, datos, rng, tam["ep"], cfg, st, None, curva, tam["cada"], Xc, yc)
        elif var == "a_oraculo":
            v = None; per = OraculoP(); th = thinker_l3(seed)
            entrenar(th, per, datos, rng, tam["ep"], {}, st, None, curva, tam["cada"])
        elif var.startswith("a_ruido"):
            acc = float(var.split("_")[-1])
            v = None; per = RuidosaP(acc, 50 + seed); th = thinker_l3(seed)
            entrenar(th, per, datos, rng, tam["ep"], {}, st, None, curva, tam["cada"])
        else:
            raise ValueError(var)
        out("f0", var, "segundos entrenamiento", round(time.time() - t0, 1), "", f"{tam['ep']} episodios")
        medir(out, "f0", var, th, v, per, datos, rng, tam, st, extra, curva, seed)
        out("f0", var, "segundos", round(time.time() - t0, 1), "", "")


def desplazamientos(conf: np.ndarray) -> np.ndarray:
    """Fracción de imágenes con símbolo s = d + k (mod 10), k = 0..9. Con el Thinker
    oráculo el código está anclado a los dígitos: un código desplazado delata soluciones
    localmente consistentes (k = 5: s(a) + s(b) = a + b mod 10; k = +-1 mezclados: los
    errores de dos ranuras se compensan)."""
    tot = conf.sum()
    return np.array([sum(conf[d, (d + k) % 10] for d in range(10)) / tot for k in range(10)])


def fusiones(conf: np.ndarray) -> int:
    """Símbolos que son el más frecuente de dos o más dígitos (clases fundidas)."""
    maj = conf.argmax(1)
    return int(sum(1 for s in np.unique(maj) if (maj == s).sum() > 1))


def medir(out, fase, var, th, v, per, datos, rng, tam, st, extra, curva, seed, congelada=True):
    """Todo lo que se mide después de entrenar: curvas, percepción, crédito, sistema
    completo (n=3 y n=1), cota pureza^6, exacta | imágenes bien según el código final y la
    descomposición (el Thinker aprendido con el código final de la percepción como
    oráculo). Con `congelada` y la variante "rep", también el control (a) de la
    percepción congelada en su código final con un Thinker nuevo."""
    for f in curva:
        for k, x in f.items():
            if k != "ep":
                out(fase, var, f"curva {k} ep={f['ep']}", x, "", "")
    if v is not None:
        mp = medir_percepcion(out, fase, var, v, Xt_(datos, seed, tam), yt_(datos, seed, tam), extra)
        pi = mp["pi"]
    else:
        pi = np.arange(10)
    if st.get("creditos"):
        out(fase, var, "créditos a la percepción", st["creditos"], "", "")
        ok = [pi[dg] == s for dg, s in st["pares"]]
        out(fase, var, "precisión del crédito (código final)", float(np.mean(ok)), 0.1, "")
        out(fase, var, "precisión del crédito (identidad)", st["creditos_bien"] / st["creditos"], 0.1, "")
        mitad = len(ok) // 2
        out(fase, var, "precisión del crédito, 2.a mitad (código final)", float(np.mean(ok[mitad:])), 0.1, "")
        if "auto" in st:
            out(fase, var, "fracción de enseñanzas que confirman lo percibido", st["auto"] / st["creditos"], "",
                "auto-confirmación")
    if hasattr(th, "abstained"):
        out(fase, var, "abstenciones Thinker", th.abstained, "", "")
    per_ev = Percepcion3(v) if v is not None else per
    m = evaluar(th, per_ev, datos, rng, 3, tam["ep_eval"])
    out(fase, var, "exacta n=3 (test)", m["exacta"], analisis()["azar_exacta_n3_mejor_constante"], "azar")
    out(fase, var, "columna n=3 (test)", m["columna"], 0.1, "azar")
    out(fase, var, "fin (acarreo) n=3 (test)", m["fin"], max(p_acarreo(3), 1 - p_acarreo(3)), "mejor constante")
    m1 = evaluar(th, per_ev, datos, rng, 1, tam["ep_eval"])
    out(fase, var, "exacta n=1 (test)", m1["exacta"], 0.1 * 0.55, "azar")
    res = dict(exacta=m["exacta"], columna=m["columna"])
    if v is not None:
        out(fase, var, "cota pureza^6 (Thinker perfecto)", mp["pureza"] ** 6, "", "")
        mc = evaluar_con_codigo(th, per_ev, datos, rng, 3, tam["ep_eval"], pi)
        out(fase, var, "exacta n=3 | imágenes bien (código)", mc["exacta_si_todas_bien"], "", "")
        out(fase, var, "fracción episodios con las 6 imágenes bien (código)", mc["todas_bien"],
            mp["pureza"] ** 6, "pureza^6")
        if not isinstance(th, ThinkerOraculo):
            mo = evaluar(th, OraculoP(pi), datos, rng, 3, tam["ep_eval"])
            out(fase, var, "Thinker con código-oráculo: exacta n=3", mo["exacta"], "", "símbolos = código final perfecto")
            out(fase, var, "Thinker con código-oráculo: columna n=3", mo["columna"], 0.1, "")
            if congelada and tam.get("congelada") and var == "rep":
                rngc = np.random.default_rng(4000 + seed)
                thc = thinker_l3(seed)
                perc = Congelada(copy.deepcopy(v))
                t1 = time.time()
                entrenar(thc, perc, datos, rngc, tam["ep"], {}, {})
                mz = evaluar(thc, Percepcion3(perc.v), datos, rngc, 3, tam["ep_eval"])
                var2 = "a_congelada"
                out(fase, var2, "segundos entrenamiento", round(time.time() - t1, 1), "", "")
                out(fase, var2, "pureza percepción (test)", mp["pureza"], 0.1, "la de rep, congelada")
                out(fase, var2, "exacta n=3 (test)", mz["exacta"], mp["pureza"] ** 6, "pureza^6")
                out(fase, var2, "columna n=3 (test)", mz["columna"], 0.1, "azar")
                mo = evaluar(thc, OraculoP(pi), datos, rngc, 3, tam["ep_eval"])
                out(fase, var2, "Thinker con código-oráculo: exacta n=3", mo["exacta"], "", "")
        out(fase, var, "clases fundidas (símbolo mayoritario de >=2 dígitos)", fusiones(mp["conf"]), 0, "")
        dsp = desplazamientos(mp["conf"])
        out(fase, var, "masa en s = d (identidad)", float(dsp[0]), 0.1, "fracción de imágenes")
        out(fase, var, "masa en s = d +- 1 (vecinos)", float(dsp[1] + dsp[9]), 0.2, "")
        out(fase, var, "masa en s = d + 5", float(dsp[5]), 0.1, "invisible para el dígito módulo 10")
        out(fase, var, "converge a una permutación (pureza - identidad)", mp["pureza"] - mp["identidad"], "",
            "alto: código consistente pero permutado")
        res.update(pureza=mp["pureza"], identidad=mp["identidad"], pi=pi)
    else:
        mo = evaluar(th, OraculoP(), datos, rng, 3, tam["ep_eval"])
        out(fase, var, "Thinker con oráculo: exacta n=3", mo["exacta"], "", "")
    return res


_XT: dict = {}


def Xt_(datos, seed, tam):
    """Muestra de test para medir la percepción (la misma que la ronda 2 por semilla)."""
    k = (seed, tam["n_test_pureza"], SPLIT_EVAL[0])
    if k not in _XT:
        if SPLIT_EVAL[0] == "test":
            _XT[k] = ei.muestra_test(datos, np.random.default_rng(2000 + seed), tam["n_test_pureza"])
        else:
            r = np.random.default_rng(2000 + seed)
            idx = r.choice(len(datos.y["pre"]), size=tam["n_test_pureza"], replace=False)
            _XT[k] = (datos.X["pre"][idx], datos.y["pre"][idx])
    return _XT[k][0]


def yt_(datos, seed, tam):
    Xt_(datos, seed, tam)
    return _XT[(seed, tam["n_test_pureza"], SPLIT_EVAL[0])][1]


# ====================================================================== F1
def construir(var_cfg, seed, tam):
    """Percepción (desde cero, con homeostasis), razonador y cfg del episodio para una
    variante de F1-F5. var_cfg: dict(thinker="aprendido"|"oraculo", posterior=dict(...) o
    None, cred2=bool, refuerzo=bool, thinker_kw=dict, cortex_kw=dict)."""
    v = ei.Vista(seed, homeostasis=tam["homeostasis"])
    if var_cfg.get("cortex_kw") or "h" in var_cfg or var_cfg.get("config_kw"):
        from exp_vision import CONFIGS
        from bioelastic import Cortex
        v.net = Cortex((28, 28, 1), 10, seed=seed, homeostasis=var_cfg.get("h", tam["homeostasis"]),
                       **dict(CONFIGS[v.config], **var_cfg.get("config_kw", {})), **var_cfg.get("cortex_kw", {}))
    per = Percepcion3(v, k=2)
    if var_cfg.get("thinker") == "oraculo":
        th = ThinkerOraculo()
    elif var_cfg.get("thinker") == "cadena":
        from bioelastic.posterior import ChainRule
        th = ChainRule(seed=seed, **(var_cfg.get("regla_kw") or {}))
    else:
        th = thinker_l3(seed, **var_cfg.get("thinker_kw", {}))
    cfg = dict(refuerzo=var_cfg.get("refuerzo", False), cred=var_cfg.get("cred2", False))
    if var_cfg.get("cadena") is not None:
        return v, per, th, dict(posterior=CreditoCadena(seed=seed, **var_cfg["cadena"]))
    if var_cfg.get("posterior") is not None:
        kw = dict(var_cfg["posterior"])
        cp = CreditoPosterior(refuerzo=var_cfg.get("refuerzo", False), fin=kw.pop("fin", False), seed=seed, **kw)
        cfg = dict(posterior=cp)
    return v, per, th, cfg


VARIANTES_F1 = {
    # sólo el posterior con lo que el Thinker da por establecido (sin nada que rompa la simetría)
    "post": dict(posterior=dict(k=10)),
    # + R-STDP de la ronda 2 como fuente de asimetría inicial
    "post+R": dict(posterior=dict(k=10), refuerzo=True),
    # + co-ocurrencia trans-situacional (vida media 2000 columnas) y consolidación en el Thinker
    "post+tabla+consol+R": dict(posterior=dict(k=10, half_life=2000, consolidate=(3, 0.8)), refuerzo=True),
    # + ACh (gamma según la fiabilidad observada) y aprender sólo lo que la evidencia movió
    "completo": dict(posterior=dict(k=10, half_life=2000, consolidate=(3, 0.8), min_gain=1.5,
                                    reliability=dict(rate=0.01)), refuerzo=True),
    # objetivo muestreado del posterior (EM estocástico) en vez del máximo
    "muestreo": dict(posterior=dict(k=10, half_life=2000, consolidate=(3, 0.8), mode="sample")),
    # Thinker oráculo: el crédito posterior con la verosimilitud exacta
    "b_post+R": dict(thinker="oraculo", posterior=dict(k=10), refuerzo=True),
    "b_post_muestreo": dict(thinker="oraculo", posterior=dict(k=10, mode="sample")),
    # ídem, con la última columna explicada también por el acarreo final (prueba de que
    # el código s = d + 5 sobrevive porque el acarreo no le da crédito a la percepción)
    "b_post_muestreo+fin": dict(thinker="oraculo", posterior=dict(k=10, mode="sample", fin=True)),
}


VARIANTES_F1C = {   # h = 0: sin homeostasis de salida (elegido en validación con MNIST-addition)
    "conocida": dict(thinker="oraculo", h=0.0, cadena=dict(regla="conocida", modo="blando")),
    "conocida_muestreo": dict(thinker="oraculo", h=0.0, cadena=dict(regla="conocida", modo="muestreo")),
    "aprendida_muestreo": dict(thinker="cadena", h=0.0, cadena=dict(regla="aprendida", modo="muestreo")),
    "aprendida_blando": dict(thinker="cadena", h=0.0, cadena=dict(regla="aprendida", modo="blando")),
    # + prior de equipartición (beta = 1, elegido en el modelo tabular: 4/4 semillas sin fusiones)
    "aprendida_eq": dict(thinker="cadena", h=0.0, cadena=dict(regla="aprendida", modo="muestreo", equiparticion=1.0)),
    # regla aprendida con una percepción sin memoria de ejemplares (sin área de asociación:
    # los símbolos salen de la proyección lineal sobre los rasgos de V1 y del lector local):
    # menos unidades independientes que coordinar en un código arbitrario
    "aprendida_eq_proy": dict(thinker="cadena", h=0.0, config_kw=dict(heads=[], replay=0.0),
                              cadena=dict(regla="aprendida", modo="muestreo", equiparticion=1.0)),
    "aprendida_muestreo_proy": dict(thinker="cadena", h=0.0, config_kw=dict(heads=[], replay=0.0),
                                    cadena=dict(regla="aprendida", modo="muestreo")),
    # regla conocida con la evidencia perceptiva pesada por su fiabilidad medida (contra los atajos)
    "conocida_fiab": dict(thinker="oraculo", h=0.0, cadena=dict(regla="conocida", modo="blando", fiabilidad=dict())),
    # piso de gamma más alto: con dígitos módulo 10 la regla sola casi no da señal por imagen
    # (I(a; d) = 0), así que con la compuerta cerrada la percepción no arranca
    "conocida_fiab_piso": dict(thinker="oraculo", h=0.0, cadena=dict(regla="conocida", modo="blando",
                                                                     fiabilidad=dict(lo=0.3, k=1.0))),
}


def fase_f1c(seed, tam, out, variante, extra):
    """F1 con posterior conjunto por episodio (ver `CreditoCadena`)."""
    if CURRICULO[0] == "aleatorio":
        return fase_f1(seed, tam, out, variante, extra, VARIANTES_F1C, "f1c")
    datos, Xt, yt, Xc, yc = preparar(seed, tam)
    from bioelastic.curriculum import Curriculum, sum_stages
    for var in (list(VARIANTES_F1C) if variante is None else [variante]):
        rng = np.random.default_rng(2000 + seed)
        ei.muestra_test(datos, rng, tam["n_test_pureza"])
        v, per, th, cfg = construir(VARIANTES_F1C[var], seed, tam)
        cu = Curriculum(sum_stages(CURRICULO[0]), tam["columnas"])
        t0 = time.time(); st: dict = {}; curva: list = []
        entrenar_curriculo(th, per, datos, rng, cu, cfg, st, curva, tam["cada_col"], Xc, yc)
        nom = f"{var} | {CURRICULO[0]}"
        out("f1c", nom, "segundos entrenamiento", round(time.time() - t0, 1), "", f"{tam['columnas']} columnas")
        medir(out, "f1c", nom, th, v, per, datos, rng, tam, st, extra, curva, seed, congelada=False)


VARIANTES_F1U = {
    # etapa 1 sin etiquetas (categorías visuales: prototipos del área de asociación y rasgos
    # de V1), etapa 2-3 como aprendida_eq (posterior conjunto + equipartición + muestreo)
    # con la estructura de la memoria congelada: se ligan los prototipos que ya hay
    "u_cabeza": dict(thinker="cadena", h=0.0, cadena=dict(regla="aprendida", modo="muestreo", equiparticion=1.0)),
    # categorías más gruesas en la etapa 1 (umbral de reclutamiento más bajo en la memoria:
    # menos prototipos, K efectivo menor, a costa de pureza)
    "u_cabeza_gruesa": dict(thinker="cadena", h=0.0, config_kw=dict(heads=[dict(source="input", layer=dict(theta=0.55))]),
                            cadena=dict(regla="aprendida", modo="muestreo", equiparticion=1.0)),
    # ídem sin memoria de ejemplares: rasgos de V1 sin etiquetas y lectura lineal (proyección)
    "u_proy": dict(thinker="cadena", h=0.0, config_kw=dict(heads=[], replay=0.0),
                   cadena=dict(regla="aprendida", modo="muestreo", equiparticion=1.0)),
}


def pureza_no_supervisada(net, E, X, y) -> dict:
    """Pureza de las categorías que se formaron sin etiquetas: cada imagen cae en la neurona
    ganadora del área de asociación; cada neurona se rotula con el dígito mayoritario de las
    imágenes de posición par que gana y se mide en las de posición impar (las etiquetas sólo
    miden, como en Diehl y Cook, 2015; una neurona sin rótulo cuenta como error). Devuelve
    la pureza y cuántas neuronas ganan alguna imagen."""
    if not net.heads:
        return dict(pureza=float("nan"), neuronas=0)
    w = []
    for x in X:
        net.forward(E(x))
        st = net.heads[0]._last
        w.append(int(st.v.winners[0]) if len(st.v.winners) else -1)
    w = np.array(w); y = np.asarray(y).astype(int)
    a, b = slice(0, None, 2), slice(1, None, 2)           # rotular con las pares, medir con las impares
    rot = {int(j): int(np.bincount(y[a][w[a] == j], minlength=10).argmax()) for j in np.unique(w[a])}
    ok = np.array([rot.get(int(j), -1) == t for j, t in zip(w[b], y[b])])
    return dict(pureza=float(ok.mean()), neuronas=int(len(np.unique(w))))


def fase_f1u(seed, tam, out, variante, extra):
    """L3 con regla desconocida en etapas (ver VARIANTES_F1U): 1) `tam["n_nosup"]` imágenes
    de train sin etiquetas (Cortex.observe: neurogénesis por novedad y STDP en la memoria,
    rasgos de V1 sin error local); se mide la pureza de las categorías formadas; 2-3) L3 desde
    ahí como aprendida_eq, con la neurogénesis de la memoria apagada."""
    from exp_vision import encoder
    datos, Xt, yt, Xc, yc = preparar(seed, tam)
    E = encoder()
    for var in (list(VARIANTES_F1U) if variante is None else [variante]):
        rng = np.random.default_rng(2000 + seed)
        ei.muestra_test(datos, rng, tam["n_test_pureza"])
        v, per, th, cfg = construir(VARIANTES_F1U[var], seed, tam)
        t0 = time.time()
        idx = np.random.default_rng(5000 + seed).permutation(len(datos.y["train"]))[:tam["n_nosup"]]
        for i in idx:
            v.net.observe(E(datos.X["train"][i]))
        pz = pureza_no_supervisada(v.net, E, Xt[:1000], yt[:1000])
        out("f1u", var, "etapa 1: segundos", round(time.time() - t0, 1), "", f"{tam['n_nosup']} imágenes sin etiquetas")
        out("f1u", var, "etapa 1: pureza de las categorías sin etiquetas", pz["pureza"], 0.1, "neurona ganadora rotulada por mayoría")
        out("f1u", var, "etapa 1: neuronas que ganan alguna imagen", pz["neuronas"], 10, "")
        for h in v.net.heads:
            h.layer.neurogenesis = False
        st: dict = {}; curva: list = []
        entrenar(th, per, datos, rng, tam["ep"], cfg, st, None, curva, tam["cada"], Xc, yc)
        out("f1u", var, "etapas 2-3: segundos", round(time.time() - t0, 1), "", f"{tam['ep']} episodios")
        medir(out, "f1u", var, th, v, per, datos, rng, tam, st, extra, curva, seed, congelada=False)


def fase_f1up(seed, tam, out, variante, extra):
    """Etapa 1 sola: cómo cambia la pureza de las categorías sin etiquetas con la cantidad de
    imágenes sin etiquetas (2 500, 10 000, 40 000; acumulativo, una pasada) y con el umbral de
    reclutamiento de la memoria (theta 0.7: por defecto, muchos prototipos; 0.55: menos).
    La pureza (rotular con la mitad de las imágenes de medición, medir con la otra) es la cota
    de lo que una ligadura perfecta de prototipos a símbolos podría dar por imagen."""
    from exp_vision import CONFIGS, encoder
    from bioelastic import Cortex
    datos, Xt, yt, Xc, yc = preparar(seed, tam)
    E = encoder()
    marcas = tam["marcas_nosup"]
    for th in ((0.7, 0.55) if variante is None else (float(variante),)):
        net = Cortex((28, 28, 1), 10, seed=seed,
                     **dict(CONFIGS["depth2"], heads=[dict(source="input", layer=dict(theta=th))]))
        idx = np.random.default_rng(5000 + seed).permutation(len(datos.y["train"]))[:max(marcas)]
        t0 = time.time(); hecho = 0
        for m in marcas:
            for i in idx[hecho:m]:
                net.observe(E(datos.X["train"][i]))
            hecho = m
            pz = pureza_no_supervisada(net, E, Xt[:1000], yt[:1000])
            var = f"theta={th}"
            out("f1up", var, f"pureza sin etiquetas n={m}", pz["pureza"], 0.1, "cota por imagen de una ligadura perfecta")
            out("f1up", var, f"neuronas ganadoras n={m}", pz["neuronas"], 10, f"{net.heads[0].layer.n} creadas")
            out("f1up", var, f"segundos n={m}", round(time.time() - t0, 1), "", "")


def fase_f1(seed, tam, out, variante, extra, variantes=None, fase="f1"):
    datos = ei.Digitos(tam["n_pre"])
    Xt, yt = Xt_(datos, seed, tam), yt_(datos, seed, tam)
    Xc, yc = Xt[:tam["n_curva"]], yt[:tam["n_curva"]]
    variantes = variantes or VARIANTES_F1
    for var in (tam.get(fase) or list(variantes)) if variante is None else [variante]:
        vc = variantes[var]
        rng = np.random.default_rng(2000 + seed)
        ei.muestra_test(datos, rng, tam["n_test_pureza"])
        v, per, th, cfg = construir(vc, seed, tam)
        t0 = time.time(); st: dict = {}; curva: list = []
        entrenar(th, per, datos, rng, tam["ep"], cfg, st, None, curva, tam["cada"], Xc, yc)
        out(fase, var, "segundos entrenamiento", round(time.time() - t0, 1), "", f"{tam['ep']} episodios")
        if "posterior" in cfg:
            for k, x in cfg["posterior"].n.items():
                out(fase, var, f"posterior: {k}", x, "", "")
        medir(out, fase, var, th, v, per, datos, rng, tam, st, extra, curva, seed, congelada=False)
        out(fase, var, "segundos", round(time.time() - t0, 1), "", "")


# ====================================================================== F0s: modelo tabular del crédito
def tabular(regla: str, curriculo: str, seed: int, columnas: int = 10000, lr: float = 0.5,
            tau: float = 0.5) -> dict:
    """Modelo mínimo del problema de crédito de L3, sin imágenes: la percepción de cada
    dígito es una tabla de preferencias W[d] (softmax sobre 10 símbolos, con un poco de
    ruido inicial para romper la simetría) y el razonamiento es la suma correcta y fija
    (Thinker oráculo) sobre lo percibido, con el acarreo que sale de lo percibido. La única
    señal es el dígito de cada columna (episodios de 1 a 4 cifras, más el acarreo final).

    regla: "refuerzo"  REINFORCE por ranura con la recompensa +-1 de la columna (R-STDP)
           "map"       crédito posterior exacto, se enseña el máximo de la marginal si >= tau
           "muestreo"  crédito posterior exacto, se enseña una muestra de la marginal
           "muestreo+fin"  ídem, y la última columna también se explica con el acarreo
                       final observado (la columna END): el único dato que distingue la
                       identidad del código s = d + 5
    curriculo: "aleatorio" o "mas0" (ver `bioelastic.curriculum`)."""
    from bioelastic.curriculum import Curriculum, sample_sum, sum_stages
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, (10, 10))
    cu = Curriculum(sum_stages(curriculo), columnas)
    base = 0.0; usado = 0; hits = []

    def P(d):
        z = W[d] - W[d].max(); e = np.exp(z)
        return e / e.sum()
    while usado < columnas:
        A, B = sample_sum(cu.params(usado), rng)
        c_true = 0; c_perc = 0
        for i, (a, b) in enumerate(zip(A, B)):
            pa, pb = P(a), P(b)
            sa, sb = int(rng.choice(10, p=pa)), int(rng.choice(10, p=pb))
            y = (a + b + c_true) % 10; c_true = (a + b + c_true) // 10
            out = (sa + sb + c_perc) % 10
            r = 1.0 if out == y else -1.0
            hits.append(out == y)
            if regla == "refuerzo":
                for d, s_, p in ((a, sa, pa), (b, sb, pb)):
                    g = -p; g[s_] += 1.0
                    W[d] += lr * (r - base) * g
                base += 0.02 * (r - base)
            else:
                L = np.array([[(x + z + c_perc) % 10 == y for z in range(10)] for x in range(10)], dtype=float)
                if regla == "muestreo+fin" and i == len(A) - 1:   # el acarreo final se observa
                    L *= np.array([[(x + z + c_perc >= 10) == bool(c_true) for z in range(10)]
                                   for x in range(10)], dtype=float)
                q = pa[:, None] * pb[None, :] * L
                q = q / q.sum() if q.sum() > 0 else pa[:, None] * pb[None, :]
                for d, qm in ((a, q.sum(1)), (b, q.sum(0))):
                    if regla == "map":
                        j = int(np.argmax(qm))
                        if qm[j] < tau:
                            continue
                    else:
                        j = int(rng.choice(10, p=qm / qm.sum()))
                    g = -P(d); g[j] += 1.0
                    W[d] += lr * g
            c_perc = (sa + sb + c_perc) // 10
            usado += 1
    code = W.argmax(1)
    shift = np.bincount((code - np.arange(10)) % 10, minlength=10) / 10.0
    return dict(identidad=float(shift[0]), vecinos=float(shift[1] + shift[9]), mas5=float(shift[5]),
                biyeccion=int(len(set(code.tolist())) == 10), columna_final=float(np.mean(hits[-1000:])),
                codigo=code.tolist())


def tabular_conjunto(curriculo: str, seed: int, columnas: int = 10000, lr: float = 0.5, alpha: float = 0.1,
                     vida: float = 1400.0, modo: str = "muestreo") -> dict:
    """Modelo tabular de la co-adaptación (L2: una columna, sin acarreo): la percepción es
    una tabla W[d] como en `tabular` y el conocimiento es una tabla de co-ocurrencia
    T[s_a, s_b] -> respuesta con decaimiento (vida media `vida` columnas), los dos desde
    cero. Responde lo más frecuente en T para lo percibido. Crédito: posterior
    q ∝ p_a p_b P_T(objetivo | s_a, s_b); T suma q (trans-situacional) y la percepción
    aprende una muestra (o el máximo) de cada marginal. Mide la pureza de la percepción
    (mejor biyección: cualquier código consistente sirve) y el acierto final."""
    from scipy.optimize import linear_sum_assignment
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, (10, 10)); T = np.zeros((10, 10, 10))
    dec = 0.5 ** (1.0 / vida)

    def P(d):
        e = np.exp(W[d] - W[d].max())
        return e / e.sum()
    hits = []
    for t in range(columnas):
        a = int(rng.integers(10))
        b = 0 if (curriculo == "mas0" and t < 0.15 * columnas) else int(rng.integers(10))
        y = (a + b) % 10
        pa, pb = P(a), P(b)
        sa, sb = int(rng.choice(10, p=pa)), int(rng.choice(10, p=pb))
        out = int(np.argmax(T[sa, sb])) if T[sa, sb].sum() > 0 else int(rng.integers(10))
        hits.append(out == y)
        q = pa[:, None] * pb[None, :] * (T[:, :, y] + alpha) / (T.sum(2) + 10 * alpha)
        q /= q.sum()
        T *= dec; T[:, :, y] += q
        for d, qm in ((a, q.sum(1)), (b, q.sum(0))):
            j = int(rng.choice(10, p=qm)) if modo == "muestreo" else int(np.argmax(qm))
            g = -P(d); g[j] += 1.0
            W[d] += lr * g
    C = np.array([P(d) for d in range(10)])
    r, c = linear_sum_assignment(-C)
    return dict(pureza=float(C[r, c].sum() / 10), columna_final=float(np.mean(hits[-1000:])))


def tabular_cadena(conocida: bool, curriculo: str, seed: int, columnas: int = 10000, lr: float = 0.5,
                   half_life: float = 0.0, modo: str = "blando", balance: float = 0.0) -> dict:
    """Modelo tabular del crédito conjunto por episodio (F1): la percepción es una tabla
    W[d] (softmax) y aprende con objetivo blando (W[d] += lr (q - p), la forma tabular del
    tercer factor q - p); el posterior es el conjunto sobre todo el episodio (dígitos y
    acarreos, `chain_posterior`). Regla conocida (suma) o aprendida a la vez
    (`ChainRule`, acarreo latente). Mide la pureza (mejor biyección), la identidad y el
    acierto por columna en las últimas 1000 columnas (test-then-train)."""
    from scipy.optimize import linear_sum_assignment
    from bioelastic.curriculum import Curriculum, sample_sum, sum_stages
    from bioelastic.posterior import ChainRule, chain_posterior, known_sum_factor
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, (10, 10))
    regla = None if conocida else ChainRule(seed=seed, half_life=half_life)
    cu = Curriculum(sum_stages(curriculo), columnas)
    usado = 0; hits = []
    from bioelastic.posterior import Equipartition
    eq = Equipartition(10, balance) if balance > 0 else None

    def P(d):
        e = np.exp(W[d] - W[d].max())
        return e / e.sum()
    while usado < columnas:
        A, B = sample_sum(cu.params(usado), rng)
        pa = [P(a) for a in A]; pb = [P(b) for b in B]
        if eq is not None:
            pa = [eq(x) for x in pa]; pb = [eq(x) for x in pb]
        sa = [int(rng.choice(10, p=x)) for x in pa]; sb = [int(rng.choice(10, p=x)) for x in pb]
        digs = []; c = 0
        for a, b in zip(A, B):
            s_ = int(a) + int(b) + c; digs.append(s_ % 10); c = s_ // 10
        # respuestas (test-then-train)
        if conocida:
            cp = 0
            for x, z, d in zip(sa, sb, digs):
                s_ = x + z + cp; hits.append(s_ % 10 == d); cp = s_ // 10
            qa, qb, _, _, _ = chain_posterior(pa, pb, [known_sum_factor(d) for d in digs], np.eye(2)[c])
        else:
            regla.reset()
            for x, z, d in zip(sa, sb, digs):
                hits.append(regla.answer((x, 10 + z)) == d)
            qa, qb, _, _, _ = regla.learn(pa, pb, digs, c)
        for d, q in list(zip(A, qa)) + list(zip(B, qb)):
            if modo == "muestreo":                  # EM estocástico: un objetivo duro muestreado de q
                q = np.eye(10)[int(rng.choice(10, p=q / q.sum()))]
            W[int(d)] += lr * (q - P(int(d)))
        usado += len(A)
    C = np.array([P(d) for d in range(10)])
    r, cc = linear_sum_assignment(-C)
    return dict(pureza=float(C[r, cc].sum() / 10), identidad=float(np.mean(C.argmax(1) == np.arange(10))),
                columna_final=float(np.mean(hits[-1000:])))


def tabular_protos(K: int, seed: int, columnas: int = 10000, lr: float = 0.5, balance: float = 1.0) -> dict:
    """Como `tabular_cadena` con la regla aprendida, objetivo muestreado y equipartición,
    pero con K prototipos independientes por dígito (cada ejemplar cae en uno al azar y
    cada prototipo tiene su propia tabla de símbolos): el modelo mínimo de una percepción
    que memoriza ejemplares. Sin ancla, todos los prototipos de un dígito tienen que caer
    en el mismo símbolo arbitrario y la única presión es la estadística de las respuestas."""
    from scipy.optimize import linear_sum_assignment
    from bioelastic.posterior import ChainRule, Equipartition
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, (10, K, 10))
    regla = ChainRule(seed=seed); eq = Equipartition(10, balance) if balance > 0 else None

    def P(d, j):
        e = np.exp(W[d, j] - W[d, j].max())
        return e / e.sum()
    usado = 0; hits = []
    while usado < columnas:
        n = int(rng.integers(1, 5)); A = rng.integers(10, size=n); B = rng.integers(10, size=n)
        JA = rng.integers(K, size=n); JB = rng.integers(K, size=n)
        pa = [P(a, j) for a, j in zip(A, JA)]; pb = [P(b, j) for b, j in zip(B, JB)]
        sa = [int(rng.choice(10, p=x)) for x in pa]; sb = [int(rng.choice(10, p=x)) for x in pb]
        digs = []; c = 0
        for a, b in zip(A, B):
            s_ = int(a) + int(b) + c; digs.append(s_ % 10); c = s_ // 10
        regla.reset()
        for x, z, d in zip(sa, sb, digs):
            hits.append(regla.answer((x, 10 + z)) == d)
        qa, qb, _, _, _ = regla.learn([eq(x) for x in pa] if eq else pa, [eq(x) for x in pb] if eq else pb, digs, c)
        for d, j, q in list(zip(A, JA, qa)) + list(zip(B, JB, qb)):
            t = np.eye(10)[int(rng.choice(10, p=q / q.sum()))]
            W[d, j] += lr * (t - P(d, j))
        usado += n
    C = np.zeros((10, 10))
    for d in range(10):
        for j in range(K):
            C[d] += P(d, j) / K
    r, cc = linear_sum_assignment(-C)
    return dict(pureza=float(C[r, cc].sum() / 10), columna_final=float(np.mean(hits[-1000:])))


def fase_f0s(seed, tam, out, variante, extra):
    """F0s: el modelo tabular de `tabular`, 20 semillas por condición (por cada semilla
    pedida se corren 20 sub-semillas; se reporta la media y la fracción que termina en la
    identidad, en +5 o en un código de desplazamientos que se compensan). Con
    `variante="conjunto"`, el modelo de co-adaptación `tabular_conjunto`; con "fin", el
    posterior que también usa el acarreo final."""
    if variante and variante.startswith("protos"):   # escalado: prototipos por dígito x presupuesto
        base = tam["col_tabular"]
        Ks = (1, 3, 10) if variante == "protos" else (int(variante[len("protos"):]),)
        for K in Ks:
            for mult, n_sub in ((1, 6), (4, 6), (16, 3)):
                if mult == 16 and K == 1:
                    continue
                R = [tabular_protos(K, 100 * seed + k, base * mult) for k in range(n_sub)]
                var = f"protos K={K} | {base * mult} columnas"
                for m in ("pureza", "columna_final"):
                    v = np.array([r[m] for r in R])
                    out("f0s", var, f"{m} (media de {n_sub})", float(v.mean()), float(v.std()), "desvío en referencia")
        return
    if variante == "cadena":
        conds = [("conocida blando", True, "blando", 0.0), ("aprendida blando", False, "blando", 0.0),
                 ("aprendida muestreo", False, "muestreo", 0.0), ("aprendida muestreo+equipartición", False, "muestreo", 1.0)]
        for nombre, conocida, modo, bal in conds:
            for cur in ("aleatorio", "mas0"):
                var = f"cadena {nombre} | {cur}"
                R = [tabular_cadena(conocida, cur, 100 * seed + k, tam["col_tabular"], modo=modo, balance=bal)
                     for k in range(20)]
                for m in ("pureza", "identidad", "columna_final"):
                    v = np.array([r[m] for r in R])
                    out("f0s", var, f"{m} (media de 20)", float(v.mean()), float(v.std()), "desvío en referencia")
                out("f0s", var, "fracción con pureza >= 0.99", float(np.mean([r["pureza"] >= 0.99 for r in R])), "", "")
        return
    if variante == "conjunto":
        for modo in ("muestreo", "map"):
            for cur in ("aleatorio", "mas0"):
                var = f"conjunto L2 {modo} | {cur}"
                R = [tabular_conjunto(cur, 100 * seed + k, tam["col_tabular"], modo=modo) for k in range(20)]
                for m in ("pureza", "columna_final"):
                    v = np.array([r[m] for r in R])
                    out("f0s", var, f"{m} (media de 20)", float(v.mean()), float(v.std()), "desvío en referencia")
                out("f0s", var, "fracción con pureza >= 0.9", float(np.mean([r["pureza"] >= 0.9 for r in R])), "", "")
        return
    reglas = ("muestreo+fin",) if variante == "fin" else ("refuerzo", "map", "muestreo")
    for regla in reglas:
        for cur in ("aleatorio", "mas0"):
            var = f"{regla} | {cur}"
            R = [tabular(regla, cur, 100 * seed + k, tam["col_tabular"]) for k in range(20)]
            for m in ("identidad", "vecinos", "mas5", "biyeccion", "columna_final"):
                out("f0s", var, f"{m} (media de 20)", float(np.mean([r[m] for r in R])), "", "")
            out("f0s", var, "fracción que termina en la identidad", float(np.mean([r["identidad"] == 1.0 for r in R])), "", "")
            out("f0s", var, "fracción que termina en +5", float(np.mean([r["mas5"] == 1.0 for r in R])), "", "")
            out("f0s", var, "fracción con >= 60 % de dígitos en d+-1", float(np.mean([r["vecinos"] >= 0.6 for r in R])), "", "")
            extra[var] = [r["codigo"] for r in R]


# ====================================================================== sistemas para F2-F5
SISTEMAS = {
    "rep": dict(cred2=True, refuerzo=True),                      # cred2+H de la ronda 2
    "oraculoR": dict(thinker="oraculo", refuerzo=True),          # Thinker oráculo + R-STDP (control b de F0)
    **{k: v for k, v in VARIANTES_F1.items() if not k.startswith("b_")},
    "oraculo_post_fin": VARIANTES_F1["b_post_muestreo+fin"],      # Thinker oráculo + posterior con acarreo final
}


def _sistemas_cadena():  # noqa: D401
    """Los sistemas de F1 con posterior conjunto (se definen más abajo)."""
    SISTEMAS.update({f"cadena_{k}": v for k, v in VARIANTES_F1C.items()})
SISTEMA = ["rep"]                    # el que usan F2, F4 y F5 (--sistema)
CURRICULO = ["aleatorio"]            # con qué currículo se entrena en F2 y F5 (--curriculo)


def gen_curriculo(kind: str, budget_cols: float, filtro=None):
    """Generador de episodios (para `entrenar`) que sigue un currículo de
    `bioelastic.curriculum` por columnas usadas; `filtro(a, b) -> bool` vuelve a sortear
    las columnas que no lo cumplen (pares excluidos, dígito ausente)."""
    from bioelastic.curriculum import Curriculum, sample_sum, sum_stages
    cu = Curriculum(sum_stages(kind), budget_cols)
    usado = [0]

    def g(e, rng):
        prm = cu.params(usado[0])
        a, b = sample_sum(prm, rng)
        if filtro is not None:
            for i in range(len(a)):
                for _ in range(1000):                    # (si la etapa no deja ninguno, queda el último)
                    if filtro(int(a[i]), int(b[i])):
                        break
                    a[i] = prm["a"][int(rng.integers(len(prm["a"])))]
                    b[i] = prm["b"][int(rng.integers(len(prm["b"])))]
        usado[0] += len(a)
        return a, b
    return g


def sistema(extra=None) -> dict:
    cfg = dict(SISTEMAS[SISTEMA[0]])
    for k, v in (extra or {}).items():
        cfg[k] = dict(cfg.get(k, {}), **v) if isinstance(v, dict) else v
    return cfg


def preparar(seed, tam):
    datos = ei.Digitos(tam["n_pre"])
    Xt, yt = Xt_(datos, seed, tam), yt_(datos, seed, tam)
    return datos, Xt, yt, Xt[:tam["n_curva"]], yt[:tam["n_curva"]]


# ====================================================================== F2
def registrar_eval(th, per, datos, rng, n, episodios) -> list:
    """Evaluación sin aprender que guarda las señales internas de cada columna."""
    ei.plasticidad(th, False)
    reg: list = []
    for _ in range(episodios):
        episodio(th, per, datos, SPLIT_EVAL[0], rng, n, False, {}, {}, registro=reg)
    ei.plasticidad(th, True)
    return reg


def senales_columna(reg: list):
    """Matriz de señales por columna de dígitos (sin la del acarreo final) y aciertos."""
    filas = [r for r in reg if not r["fin"] and "ppa" in r]
    ok = np.array([r["ok"] for r in filas], dtype=bool)
    g = lambda k, d=0.0: np.array([r.get(k, d) for r in filas], dtype=np.float64)  # noqa: E731
    sig = {
        "1-p percepción (pa*pb)": 1 - g("ppa") * g("ppb"),
        "conflicto Cortex (máx a, b)": np.maximum(g("conf_cxa"), g("conf_cxb")),
        "1-acuerdo votantes (mín)": 1 - np.minimum(g("acuerdoa", 1), g("acuerdob", 1)),
        "1-p Thinker": 1 - g("pt", 1),
        "no establecido (Thinker)": 1 - g("tsure", 1),
        "conflicto percepción-Thinker": g("conflicto_pt"),
        "1-p sistema (pa*pb*pt)": 1 - g("ppa") * g("ppb") * g("pt", 1),
        "1-P(respuesta | percepción, regla)": 1 - g("p_sis", 0.5),
    }
    # confianza cruda del sistema: P(respuesta | percepción, regla) si el razonador la da;
    # si no, el producto de las probabilidades de la percepción y del Thinker
    tiene = np.array(["p_sis" in r for r in filas], dtype=bool)
    cruda = np.where(tiene, g("p_sis", 0.5), g("ppa") * g("ppb") * g("pt", 1))
    sig["1-confianza cruda del sistema"] = 1 - cruda
    feats = np.stack([np.ones(len(ok)), np.log(np.clip(cruda, 1e-4, 1 - 1e-4) / np.clip(1 - cruda, 1e-4, 1)),
                      sig["conflicto Cortex (máx a, b)"], sig["conflicto percepción-Thinker"], g("tsure", 1)], 1)
    return sig, ok, feats, filas


def calibrar(feats_tr, ok_tr, lr=0.05):
    """Calibrador del sistema (regresión logística online, en orden: prequential)."""
    from bioelastic.metacog import Calibrator
    cal = Calibrator(n_extra=3, lr=lr)
    pre = np.zeros(len(ok_tr))
    for i, (f, o) in enumerate(zip(feats_tr, ok_tr)):
        pre[i] = cal.update(f, bool(o))
    return cal, pre


VARIANTES_F2 = {
    "sin": {},
    "hiper": dict(cortex_kw=dict(hypercorrection=1.0), thinker_kw=dict(hypercorrect=1.0)),
}


def fase_f2(seed, tam, out, variante, extra):
    """Metacognición en L3: calibración (ECE) de la percepción y del sistema, detección
    del propio error antes del feedback (AUROC de cada señal interna, prequential en el
    entrenamiento y en test) y plasticidad ponderada por confianza (hipercorrección en el
    Cortex y en el Thinker), con y sin."""
    from bioelastic.metacog import auroc, ece
    datos, Xt, yt, Xc, yc = preparar(seed, tam)
    for var in (list(VARIANTES_F2) if variante is None else [variante]):
        rng = np.random.default_rng(2000 + seed)
        ei.muestra_test(datos, rng, tam["n_test_pureza"])
        v, per, th, cfg = construir(sistema(VARIANTES_F2[var]), seed, tam)
        t0 = time.time(); st: dict = {}; curva: list = []; reg: list = []
        gen = None if CURRICULO[0] == "aleatorio" else gen_curriculo(CURRICULO[0], 2.5 * tam["ep"])
        entrenar(th, per, datos, rng, tam["ep"], cfg, st, None, curva, tam["cada"], Xc, yc, registro=reg,
                 generador=gen)
        out("f2", var, "segundos entrenamiento", round(time.time() - t0, 1), "",
            f"sistema {SISTEMA[0]}, currículo {CURRICULO[0]}")
        res = medir(out, "f2", var, th, v, per, datos, rng, tam, st, extra, curva, seed, congelada=False)
        sig, ok, feats, _ = senales_columna(reg)
        cal, pre = calibrar(feats, ok)
        mitad = np.arange(len(ok)) >= len(ok) // 2
        out("f2", var, "acierto columna prequential (2.a mitad)", float(ok[mitad].mean()), 0.1, "")
        for k, sc in sig.items():
            out("f2", var, f"AUROC error prequential (2.a mitad): {k}", auroc(sc[mitad], ~ok[mitad]), 0.5, "azar")
        out("f2", var, "AUROC error prequential (2.a mitad): 1-calibrada", auroc(1 - pre[mitad], ~ok[mitad]), 0.5, "")
        e_raw, _ = ece(1 - sig["1-confianza cruda del sistema"][mitad], ok[mitad])
        e_cal, curva_cal = ece(pre[mitad], ok[mitad])
        out("f2", var, "ECE sistema cruda (prequential, 2.a mitad)", e_raw, 0.0, "P(respuesta) o pa*pb*pt")
        out("f2", var, "ECE sistema calibrada (prequential, 2.a mitad)", e_cal, 0.0, "")
        # test: señales con todo congelado (el calibrador también)
        rt = registrar_eval(th, Percepcion3(v), datos, rng, 3, tam["ep_eval"])
        sig_t, ok_t, feats_t, filas_t = senales_columna(rt)
        p_cal = np.array([cal.predict(f) for f in feats_t])
        out("f2", var, "acierto columna (test)", float(ok_t.mean()), 0.1, "")
        for k, sc in sig_t.items():
            out("f2", var, f"AUROC error test: {k}", auroc(sc, ~ok_t), 0.5, "azar")
        out("f2", var, "AUROC error test: 1-calibrada", auroc(1 - p_cal, ~ok_t), 0.5, "")
        e_raw_t, curva_raw_t = ece(1 - sig_t["1-confianza cruda del sistema"], ok_t)
        e_cal_t, curva_cal_t = ece(p_cal, ok_t)
        out("f2", var, "ECE sistema cruda (test)", e_raw_t, 0.0, "")
        out("f2", var, "ECE sistema calibrada (test)", e_cal_t, 0.0, "")
        # percepción por imagen (test): confianza vs acierto según el código final
        pi = res.get("pi", np.arange(10))
        pp = np.array([f[k] for f in filas_t for k in ("ppa", "ppb")])
        okp = np.array([f[f"s_{k}"] == pi[f[f"dig_{k}"]] for f in filas_t for k in ("a", "b")])
        e_p, curva_p = ece(pp, okp)
        out("f2", var, "ECE percepción (test, código final)", e_p, 0.0, "")
        out("f2", var, "AUROC error percepción (test): 1-p", auroc(1 - pp, ~okp), 0.5, "")
        cx = np.array([f.get(f"conf_cx{k}", 0.0) for f in filas_t for k in ("a", "b")])
        out("f2", var, "AUROC error percepción (test): conflicto Cortex", auroc(cx, ~okp), 0.5, "")
        extra[f"f2_{var}"] = dict(confiabilidad_sistema_test=curva_cal_t, confiabilidad_cruda_test=curva_raw_t,
                                  confiabilidad_percepcion_test=curva_p, calibrador=cal.w.tolist())
        out("f2", var, "segundos", round(time.time() - t0, 1), "", "")


# ====================================================================== F3
class RuidosaDeclarada(RuidosaP):
    """Símbolos con exactitud `acc` que se declaran con certeza `declarada` (mal calibrada
    si difieren): la fiabilidad que la percepción dice tener no es la que tiene."""

    def __init__(self, acc, declarada, seed):
        super().__init__(acc, seed)
        self.declarada = float(declarada)

    def ver(self, img, d, explore):
        s, _, alts = super().ver(img, d, explore)
        self.p = np.full(10, (1 - self.declarada) / 9); self.p[s] = self.declarada
        return s, self.declarada, alts

    def ensenar(self, img, sym):
        pass

    def reforzar(self, img, sym, r):
        pass


VARIANTES_F3 = {
    # Thinker con percepción simbólica ruidosa (0.8 real, 0.95 declarada) y cambio de regla
    "base": dict(),
    "ne": dict(thinker_kw=dict(neuromod=dict(slow=0.01, fast=0.1, ne_reset=3.0, window=300, warmup=300))),
    "ach": dict(posterior=dict(k=10, reliability=dict(rate=0.01))),
    "ach+ne": dict(posterior=dict(k=10, reliability=dict(rate=0.01)),
                   thinker_kw=dict(neuromod=dict(slow=0.01, fast=0.1, ne_reset=3.0, window=300, warmup=300))),
    "post_fijo": dict(posterior=dict(k=10)),                       # control de ach: gamma = 1 fijo
}


def recuperacion(cols: np.ndarray, t0: int, nivel: float, win: int) -> int:
    """Columnas desde `t0` hasta que el acierto en una ventana de `win` llega a `nivel`."""
    seg = np.asarray(cols[t0:], dtype=np.float64)
    if len(seg) < win:
        return len(seg)
    run = np.convolve(seg, np.ones(win) / win, mode="valid")
    k = np.flatnonzero(run >= nivel)
    return int(k[0] + win) if len(k) else len(seg)


def fase_f3(seed, tam, out, variante, extra):
    """Incertidumbre esperada (ACh: el Thinker descuenta la evidencia perceptiva según su
    fiabilidad, `PosteriorCredit(reliability=...)`) e inesperada (NE: un salto de la
    sorpresa reinicia hipótesis, `Thinker(neuromod=...)`) con un cambio de regla a mitad
    del flujo: el dígito pasa a ser (a + b + acarreo + 3) mod 10 y después vuelve.
    Percepción simbólica con exactitud 0.8 que declara 0.95 (sin imágenes: barato)."""
    datos = ei.Digitos(tam["n_pre"])
    E = tam["ep_f3"]
    for var in (list(VARIANTES_F3) if variante is None else [variante]):
        vc = VARIANTES_F3[var]
        rng = np.random.default_rng(7000 + seed)
        per = RuidosaDeclarada(0.8, 0.95, 70 + seed)
        th = thinker_l3(seed, **vc.get("thinker_kw", {}))
        post = CreditoPosterior(seed=seed, **vc["posterior"]) if vc.get("posterior") else None
        cols: list = []; reglas: list = []; ne: list = []
        t0 = time.time()
        for fase_i, (desde, regla) in enumerate(((0, 0), (E // 2, 3), (E, 0))):
            cfg = dict(regla=regla)
            if post is not None:
                cfg["posterior"] = post
            n_ep = E // 2 if fase_i < 2 else E // 4

            def tras(e, r, regla=regla):
                cols.extend(r["columnas"][:-1]); reglas.extend([regla] * (len(r["columnas"]) - 1))
                ne.append(th.unexpected_uncertainty)
            entrenar(th, per, datos, rng, n_ep, cfg, {}, al_episodio=tras)
        cols = np.array(cols, dtype=np.float64); reglas = np.array(reglas)
        cambio = int(np.flatnonzero(reglas == 3)[0]); vuelta = int(np.flatnonzero((reglas == 0) & (np.arange(len(reglas)) > cambio))[0])
        W = 200
        pre = float(cols[cambio - 600:cambio].mean())
        post_lvl = float(cols[vuelta - 600:vuelta].mean())
        out("f3", var, "segundos", round(time.time() - t0, 1), "", f"{E + E // 4} episodios")
        out("f3", var, "acierto columna antes del cambio", pre, "", "últimas 600 columnas")
        out("f3", var, "acierto columna al final de la regla nueva", post_lvl, pre, "antes del cambio")
        out("f3", var, "caída inmediata (primeras 200 col. tras el cambio)", float(cols[cambio:cambio + W].mean()), pre, "")
        out("f3", var, "columnas hasta recuperar 0.9 x nivel previo", recuperacion(cols, cambio, 0.9 * pre, W),
            vuelta - cambio, "largo de la fase")
        out("f3", var, "al volver a la regla vieja: primeras 200 col.", float(cols[vuelta:vuelta + W].mean()), pre, "")
        out("f3", var, "al volver: columnas hasta 0.9 x nivel previo", recuperacion(cols, vuelta, 0.9 * pre, W),
            len(cols) - vuelta, "largo de la fase")
        out("f3", var, "reinicios NE", getattr(th, "resets", 0), "", "")
        if post is not None:
            out("f3", var, "gamma final (ACh)", post.pc.gamma, 1.0, "1 = cree a la percepción lo que declara")
            out("f3", var, "posterior: el Thinker no aprendió (desconfianza)", post.n["distrust"], "", "")
        # lo que queda de cada regla al final (test, percepción ruidosa y oráculo)
        for nombre, regla in (("vieja", 0), ("nueva", 3)):
            ei.plasticidad(th, False)
            r = [episodio(th, OraculoP(), datos, "test", rng, 3, False, dict(regla=regla), {}) for _ in range(tam["ep_eval"])]
            ei.plasticidad(th, True)
            out("f3", var, f"final: regla {nombre}, columna n=3 (oráculo)", float(np.mean([c for x in r for c in x["columnas"][:-1]])), "", "")
        extra[f"f3_{var}"] = dict(curva=[float(cols[i:i + 100].mean()) for i in range(0, len(cols), 100)],
                                  ne=[float(x) for x in ne[::10]], cambio=cambio, vuelta=vuelta)


VARIANTES_F3C = {
    "base": dict(),
    "ne": dict(regla_kw=dict(neuromod=dict())),
    "ne+compuerta": dict(regla_kw=dict(neuromod=dict()), compuerta_ne=3.0),
}


def fase_f3c(seed, tam, out, variante, extra):
    """F3 sobre el sistema completo con posterior conjunto y regla aprendida (ChainRule,
    percepción desde cero, objetivo muestreado con equipartición): después de `columnas` columnas la regla
    del dígito pasa a ser (a + b + acarreo + 3) mod 10 durante columnas/2 y luego vuelve
    durante columnas/4. NE (sorpresa del episodio bajo la regla actual por encima de la
    esperada) olvida la evidencia vieja de la regla; con compuerta, además, la percepción
    no aprende mientras NE está alto (cambió la regla, no lo que se ve)."""
    from bioelastic.curriculum import Curriculum, sum_stages
    datos, Xt, yt, Xc, yc = preparar(seed, tam)
    for var in (list(VARIANTES_F3C) if variante is None else [variante]):
        vc = VARIANTES_F3C[var]
        base = dict(VARIANTES_F1C["aprendida_eq"])
        base["regla_kw"] = vc.get("regla_kw", {})
        base["cadena"] = dict(base["cadena"], compuerta_ne=vc.get("compuerta_ne"))
        rng = np.random.default_rng(2000 + seed)
        ei.muestra_test(datos, rng, tam["n_test_pureza"])
        v, per, th, cfg = construir(base, seed, tam)
        cc = cfg["posterior"]
        C = tam["columnas"]
        fases = [("aleatorio", 0, C), ("aleatorio", 3, C // 2), ("aleatorio", 0, C // 4)]
        cols: list = []; marcas = []; pur = []; ne = []; ep_marcas = []
        t0 = time.time()
        for cur, regla, presup in fases:
            marcas.append(len(cols)); ep_marcas.append(len(ne))
            cu = Curriculum(sum_stages(cur), presup)
            from bioelastic.curriculum import sample_sum
            usado = 0
            while usado < presup:
                a, b = sample_sum(cu.params(usado), rng)
                r = episodio(th, per, datos, "train", rng, 0, True, dict(posterior=cc, regla=regla), {}, digitos=(a, b))
                cols.extend(r["columnas"][:-1]); usado += len(a)
                ne.append(th.unexpected_uncertainty)
            conf = v.confusion(Xc, yc)
            pur.append(ei.pureza(conf))
        cols = np.array(cols, dtype=np.float64)
        cambio, vuelta = marcas[1], marcas[2]
        W = 200
        pre = float(cols[cambio - 1000:cambio].mean())
        out("f3c", var, "segundos", round(time.time() - t0, 1), "", f"{C + C // 2 + C // 4} columnas")
        out("f3c", var, "acierto columna antes del cambio", pre, "", "últimas 1000 columnas")
        out("f3c", var, "pureza antes del cambio", pur[0], "", "")
        out("f3c", var, "pureza al final de la regla nueva", pur[1], pur[0], "antes del cambio")
        out("f3c", var, "pureza al final", pur[2], pur[0], "antes del cambio")
        out("f3c", var, "caída inmediata (primeras 200 col.)", float(cols[cambio:cambio + W].mean()), pre, "")
        out("f3c", var, "columnas hasta recuperar 0.9 x nivel previo", recuperacion(cols, cambio, 0.9 * pre, W),
            vuelta - cambio, "largo de la fase")
        out("f3c", var, "acierto al final de la regla nueva (últimas 1000)", float(cols[vuelta - 1000:vuelta].mean()), pre, "")
        out("f3c", var, "al volver a la regla vieja: primeras 200 col.", float(cols[vuelta:vuelta + W].mean()), pre, "")
        out("f3c", var, "al volver: columnas hasta 0.9 x nivel previo", recuperacion(cols, vuelta, 0.9 * pre, W),
            len(cols) - vuelta, "largo de la fase")
        out("f3c", var, "reinicios NE (olvido de la regla)", getattr(th, "resets", 0), "", "")
        out("f3c", var, "episodios con la percepción en pausa (compuerta NE)", cc.n.get("compuerta", 0), "", "")
        ne = np.array(ne)
        e1 = ep_marcas[1]
        out("f3c", var, "NE medio en la 2.a mitad de la fase 1", float(ne[e1 // 2:e1].mean()), "", "")
        out("f3c", var, "NE medio en los 300 episodios tras el cambio", float(ne[e1:e1 + 300].mean()), "", "")
        for nombre, regla in (("vieja", 0), ("nueva", 3)):
            ei.plasticidad(th, False)
            rr = [episodio(th, Percepcion3(v), datos, "test", rng, 3, False, dict(regla=regla), {})
                  for _ in range(tam["ep_eval"])]
            ei.plasticidad(th, True)
            out("f3c", var, f"final: regla {nombre}, exacta n=3 (test)", float(np.mean([x["exacta"] for x in rr])), "", "")
            out("f3c", var, f"final: regla {nombre}, columna n=3 (test)",
                float(np.mean([c for x in rr for c in x["columnas"][:-1]])), "", "")
        extra[f"f3c_{var}"] = dict(curva=[float(cols[i:i + 200].mean()) for i in range(0, len(cols), 200)],
                                   marcas=marcas, pureza=pur)


# ====================================================================== F4
def entrenar_curriculo(th, per, datos, rng, cu, cfg, st, curva, cada, Xc, yc):
    """Episodios del currículo `cu` hasta gastar su presupuesto de columnas."""
    from bioelastic.curriculum import sample_sum
    usado = 0; win = []; prox = cada; e = 0
    while usado < cu.budget:
        a, b = sample_sum(cu.params(usado), rng)
        r = episodio(th, per, datos, "train", rng, 0, True, cfg, st, digitos=(a, b))
        usado += len(a); win += r["columnas"][:-1]; e += 1
        if usado >= prox:
            fila = dict(ep=int(usado), columna=float(np.mean(win)), etapa=cu.stage(usado - 1))
            if Xc is not None and hasattr(per, "v"):
                conf = per.v.confusion(Xc, yc)
                fila.update(pureza=ei.pureza(conf), identidad=float(np.trace(conf) / conf.sum()))
            curva.append(fila); win = []; prox += cada
    return e


def fase_f4(seed, tam, out, variante, extra):
    """Currículo sólo por los datos presentados (mismo presupuesto de columnas, mismas
    etiquetas, misma arquitectura) contra orden aleatorio. Ver `bioelastic/curriculum.py`."""
    from bioelastic.curriculum import Curriculum, per_slot_information, sum_stages
    datos, Xt, yt, Xc, yc = preparar(seed, tam)
    for cur in (["aleatorio", "mas0", "conteo", "corto"] if variante is None else [variante]):
        rng = np.random.default_rng(2000 + seed)
        ei.muestra_test(datos, rng, tam["n_test_pureza"])
        v, per, th, cfg = construir(sistema(), seed, tam)
        cu = Curriculum(sum_stages(cur), tam["columnas"])
        t0 = time.time(); st: dict = {}; curva: list = []
        n_ep = entrenar_curriculo(th, per, datos, rng, cu, cfg, st, curva, tam["cada_col"], Xc, yc)
        var = f"{SISTEMA[0]} | {cur}"
        out("f4", var, "segundos entrenamiento", round(time.time() - t0, 1), "", f"{tam['columnas']} columnas")
        out("f4", var, "episodios", n_ep, "", "")
        out("f4", var, "información por ranura en la 1.a etapa (bits)",
            per_slot_information(cu.stages[0]["a"], cu.stages[0]["b"]), 0.0, "aleatorio")
        medir(out, "f4", var, th, v, per, datos, rng, tam, st, extra, curva, seed, congelada=False)
        out("f4", var, "segundos", round(time.time() - t0, 1), "", "")


# ====================================================================== F5
def pares_excluidos(seed, k=15):
    """k pares (a, b) que no aparecen nunca en el entrenamiento (composicional)."""
    r = np.random.default_rng(9000 + seed)
    todos = [(x, z) for x in range(10) for z in range(10)]
    return {todos[i] for i in r.choice(100, k, replace=False)}


def gen_sin_pares(H):
    def g(e, rng):
        n = int(rng.integers(1, 5))
        a = rng.integers(10, size=n); b = rng.integers(10, size=n)
        for i in range(n):
            while (int(a[i]), int(b[i])) in H:
                a[i], b[i] = rng.integers(10), rng.integers(10)
        return a, b
    return g


def gen_sin_digito(dig):
    def g(e, rng):
        n = int(rng.integers(1, 5))
        otros = np.array([x for x in range(10) if x != dig])
        return otros[rng.integers(9, size=n)], otros[rng.integers(9, size=n)]
    return g


def acierto_pares(th, per, datos, rng, pares, n_ep):
    """Acierto por columna (n=1, test) sobre los pares dados."""
    ei.plasticidad(th, False)
    pares = list(pares); hits = []
    for _ in range(n_ep):
        a, b = pares[int(rng.integers(len(pares)))]
        r = episodio(th, per, datos, SPLIT_EVAL[0], rng, 0, False, {}, {}, digitos=([a], [b]))
        hits.append(r["columnas"][0])
    ei.plasticidad(th, True)
    return float(np.mean(hits))


def fase_f5(seed, tam, out, variante, extra):
    """Pruebas adversariales, sobre el sistema elegido y sobre el Thinker con percepción
    oráculo (el razonamiento solo): generalización composicional a pares nunca vistos,
    un dígito que aparece tarde (con esquema vs sin esquema: Tse et al., 2007), latencia y
    costo por respuesta, y aprendizaje que no se detiene (test-then-train)."""
    datos, Xt, yt, Xc, yc = preparar(seed, tam)
    H = pares_excluidos(seed)
    vistos = [(x, z) for x in range(10) for z in range(10) if (x, z) not in H]
    pruebas = ["comp", "tardio"] if variante is None else [variante]
    for quien in ("oraculo", SISTEMA[0]):
        def nuevo():
            if quien == "oraculo":
                return None, OraculoP(), thinker_l3(seed), {}
            return construir(sistema(), seed, tam)
        for prueba in pruebas:
            var = f"{quien} | {prueba}" + (f" | {CURRICULO[0]}" if quien != "oraculo" and CURRICULO[0] != "aleatorio" else "")
            rng = np.random.default_rng(9500 + seed)
            v, per, th, cfg = nuevo()
            per_ev = Percepcion3(v) if v is not None else per
            t0 = time.time(); cols: list = []
            cur = CURRICULO[0] if quien != "oraculo" else "aleatorio"
            if prueba == "comp":
                gen = gen_sin_pares(H) if cur == "aleatorio" else \
                    gen_curriculo(cur, 2.5 * tam["ep"], lambda a, b: (a, b) not in H)
                entrenar(th, per, datos, rng, tam["ep"], cfg, {}, generador=gen,
                         al_episodio=lambda e, r: cols.extend(r["columnas"][:-1]))
                out("f5", var, "segundos", round(time.time() - t0, 1), "", "")
                a_v = acierto_pares(th, per_ev, datos, rng, vistos, tam["ep_eval"])
                a_h = acierto_pares(th, per_ev, datos, rng, H, tam["ep_eval"])
                out("f5", var, "columna n=1 pares vistos (test)", a_v, 0.1, "azar")
                out("f5", var, "columna n=1 pares nunca vistos (test)", a_h, 0.1, "azar")
                cols = np.array(cols, dtype=np.float64)
                m = len(cols) // 2
                pend = np.polyfit(np.arange(len(cols) - m), cols[m:], 1)[0] * 1000 if len(cols) > m + 2 else 0.0
                out("f5", var, "test-then-train: acierto columna último 10 %", float(cols[-len(cols) // 10:].mean()), "", "")
                out("f5", var, "test-then-train: pendiente 2.a mitad (por 1000 col.)", float(pend), 0.0, "")
                # costo y latencia por columna (sin aprender)
                ei.plasticidad(th, False)
                exp0 = getattr(th, "expanded", 0); t1 = time.time(); n_col = 0
                for _ in range(100):
                    r = episodio(th, per_ev, datos, SPLIT_EVAL[0], rng, 3, False, {}, {})
                    n_col += len(r["columnas"])
                dt = (time.time() - t1) / n_col
                ei.plasticidad(th, True)
                out("f5", var, "latencia por respuesta (ms, sin aprender)", 1000 * dt, "", "percepción + Thinker")
                out("f5", var, "nodos expandidos por respuesta (búsqueda)", (getattr(th, "expanded", 0) - exp0) / n_col, "", "")
                if v is not None:
                    stc = v.net.stats()
                    out("f5", var, "neuronas percepción", stc["neurons"], "", "")
                    out("f5", var, "sinapsis percepción", stc["synapses"], "", "")
                if hasattr(th, "say"):
                    out("f5", var, "neuronas conocimiento Thinker", th.say.layer.n, "", "")
                elif hasattr(th, "T"):
                    out("f5", var, "celdas de la regla aprendida con evidencia > 1", int((th.T > 1.0).sum()), "", "")
            else:                                         # dígito tardío: 9 ausente y después aparece
                E1, E2 = tam["ep"] * 3 // 4, tam["ep"] // 4
                gen = gen_sin_digito(9) if cur == "aleatorio" else \
                    gen_curriculo(cur, 2.5 * E1, lambda a, b: 9 not in (a, b))
                entrenar(th, per, datos, rng, E1, cfg, {}, generador=gen)
                eventos = []
                rng2 = np.random.default_rng(9600 + seed)
                for _ in range(E2):
                    n = int(rng2.integers(1, 5))
                    eventos.append((rng2.integers(10, size=n), rng2.integers(10, size=n)))
                for esquema, (vv, pp, tt, cc) in (("con esquema", (v, per, th, cfg)), ("sin esquema", nuevo())):
                    rngx = np.random.default_rng(9700 + seed)
                    con9, sin9 = [], []

                    def tras(e, r, ev=None):
                        a, b = eventos[e]
                        for i, ok in enumerate(r["columnas"][:-1]):
                            (con9 if 9 in (a[i], b[i]) else sin9).append(ok)
                    entrenar(tt, pp, datos, rngx, E2, cc, {}, generador=lambda e, r_: eventos[e], al_episodio=tras)
                    k = f"{var} | {esquema}"
                    for w in (50, 200):
                        out("f5", k, f"acierto en las primeras {w} columnas con 9", float(np.mean(con9[:w])), 0.1, "")
                    out("f5", k, "acierto en las últimas 200 columnas con 9", float(np.mean(con9[-200:])), 0.1, "")
                    out("f5", k, "acierto en columnas sin 9 (toda la fase)", float(np.mean(sin9)), 0.1, "")
                    ppe = Percepcion3(vv) if vv is not None else pp
                    out("f5", k, "test: columna n=1 con 9", acierto_pares(tt, ppe, datos, rngx,
                                                                         [(x, z) for x in range(10) for z in range(10) if 9 in (x, z)],
                                                                         tam["ep_eval"]), 0.1, "")
                out("f5", var, "segundos", round(time.time() - t0, 1), "", "")


# ====================================================================== principal
TAM = dict(
    completa=dict(n_pre=10000, n_test_pureza=2000, n_curva=500, ep=4000, cada=500, ep_eval=500,
                  homeostasis=0.02, congelada=True, ep_f3=4000, columnas=10000, cada_col=1250, col_tabular=10000,
                  n_nosup=10000, marcas_nosup=(2500, 10000, 40000),
                  f0=["rep", "a_oraculo", "a_ruido_0.65", "a_ruido_0.8", "a_ruido_0.9",
                      "b_refuerzo", "b_cred2", "c_barajadas", "rep_v02"]),
    rapida=dict(n_pre=300, n_test_pureza=100, n_curva=50, ep=30, cada=10, ep_eval=10,
                homeostasis=0.02, congelada=True, ep_f3=800, columnas=80, cada_col=20, col_tabular=300,
                n_nosup=100, marcas_nosup=(50, 100, 200),
                f0=["rep", "a_oraculo", "a_ruido_0.8", "b_refuerzo", "b_cred2", "c_barajadas"]),
)
_sistemas_cadena()
FASES = dict(f0=fase_f0, f0s=fase_f0s, f1=fase_f1, f1c=fase_f1c, f1u=fase_f1u, f1up=fase_f1up, f3c=fase_f3c, f2=fase_f2, f3=fase_f3, f4=fase_f4, f5=fase_f5)


def completo(path) -> bool:
    """¿El CSV de una corrida existe y terminó (tiene la fila "total" del final)?"""
    try:
        return any(",total," in ln for ln in path.read_text().splitlines()[-3:])
    except OSError:
        return False


def resumen():
    """Media, desvío, mínimo y máximo por (fase, variante, métrica) sobre las semillas, en
    results/integracion3_resumen.csv y un JSON por fase (con el análisis de azar e
    información mutua y, si las hay, las matrices de confusión por semilla)."""
    filas = {}
    for p in sorted(RESULTS.glob("integracion3_*_seed*.csv")):
        with p.open() as f:
            for r in csv.DictReader(f):
                try:
                    v = float(r["valor"])
                except ValueError:
                    continue
                k = (r["prueba"], r["variante"], r["metrica"])
                filas.setdefault(k, dict(v=[], s=[], ref=r["referencia"], nota=r["nota"]))
                filas[k]["v"].append(v); filas[k]["s"].append(int(r["semilla"]))
    log = CsvLog("integracion3_resumen", ["fase", "variante", "metrica", "media", "desvio", "min", "max",
                                          "semillas", "por_semilla", "referencia", "nota"])
    js: dict = {}
    for (fa, var, met), d in sorted(filas.items()):
        v = np.array(d["v"])
        log.row(fa, var, met, float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                float(v.min()), float(v.max()), len(v), " ".join(f"{x:.4f}" for x in v), d["ref"], d["nota"])
        js.setdefault(fa, {}).setdefault(var, {})[met] = dict(
            media=float(v.mean()), desvio=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            semillas=d["s"], valores=v.tolist(), referencia=d["ref"], nota=d["nota"])
    log.close()
    for fa, cont in js.items():
        extra = {}
        for p in sorted(RESULTS.glob(f"integracion3_{fa}_*seed*.json")):
            extra[p.stem] = json.loads(p.read_text())
            for k, v in extra[p.stem].items():        # desplazamientos s - d (mod 10) de cada código
                if isinstance(v, dict) and "confusion" in v:
                    v["desplazamientos"] = [round(float(x), 4) for x in desplazamientos(np.array(v["confusion"]))]
        doc = dict(fase=fa, metricas=cont, por_corrida=extra, tamanos=TAM["completa"])
        if fa == "f0":
            doc["analisis"] = analisis()
        (RESULTS / f"integracion3_{fa}.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fase", choices=list(FASES) + ["resumen", "analisis"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variante", default=None)
    ap.add_argument("--rapida", action="store_true", help="tamaños mínimos (humo), sólo imprime")
    ap.add_argument("--val", action="store_true",
                    help="evaluar en validación (train[:n_pre], que L3 no usa) y no en test: para elegir")
    ap.add_argument("--ep", type=int, default=0, help="episodios de entrenamiento (0 = los de la fase)")
    ap.add_argument("--sistema", default="rep", help="sistema de L3 para F2, F4 y F5: " + ", ".join(SISTEMAS))
    ap.add_argument("--curriculo", default="aleatorio", help="currículo con el que se entrena en F2 y F5")
    ap.add_argument("--skip-existing", action="store_true", help="no correr si el CSV de salida ya está completo")
    args = ap.parse_args()
    SISTEMA[0] = args.sistema
    CURRICULO[0] = args.curriculo
    if args.val:
        SPLIT_EVAL[0] = "pre"
    if args.ep:
        TAM["completa"]["ep"] = args.ep
    if args.fase == "resumen":
        resumen()
        return
    if args.fase == "analisis":
        print(json.dumps(analisis(), indent=1))
        return
    t0 = time.time()
    extra: dict = {}
    if args.rapida:
        FASES[args.fase](args.seed, TAM["rapida"], lambda *r: print(*r, sep="  ", flush=True), args.variante, extra)
        print("segundos", round(time.time() - t0, 1))
        return
    nombre = (f"{'val3' if args.val else 'integracion3'}_{args.fase}"
              + (f"_{args.variante.replace('+', '-')}" if args.variante else "")
              + (f"_{args.sistema.replace('+', '-')}" if args.sistema != "rep" else "")
              + (f"_{args.curriculo}" if args.curriculo != "aleatorio" else "")
              + (f"_ep{args.ep}" if args.ep else "") + f"_seed{args.seed}")
    nombre = nombre.replace("val3_", "integracion3_").replace(f"_{args.fase}", f"_{args.fase}val", 1) \
        if args.val else nombre
    if args.skip_existing and completo(RESULTS / f"{nombre}.csv"):
        print("ya está:", nombre)
        return
    log = CsvLog(nombre, HEADER)
    suf = "val" if args.val else ""
    FASES[args.fase](args.seed, TAM["completa"], lambda *r: log.row(r[0] + suf, r[1], args.seed, *r[2:]),
                     args.variante, extra)
    log.row(args.fase + suf, "total", args.seed, "segundos", round(time.time() - t0, 1), "", "")
    log.close()
    if extra:
        (RESULTS / f"{nombre}.json").write_text(json.dumps(extra, ensure_ascii=False))


if __name__ == "__main__":
    main()
