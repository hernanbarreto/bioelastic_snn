"""Pruebas de integración (equipo rojo): percepción real, razonamiento y memoria juntos.

Hasta acá cada pieza se probó por separado: el Cortex sobre imágenes o audio con
etiquetas, y el Thinker sobre símbolos one-hot perfectos. Acá se los obliga a funcionar
juntos y se mide dónde se rompe. La interfaz entre los dos es la única que ofrece la API
pública del Thinker: símbolos discretos (la decisión del Cortex es el símbolo que el
Thinker recibe).

Toda evaluación usa ejemplares del split oficial de test (MNIST t10k; FSDD grabaciones
0-4 de cada locutor y dígito) que la red nunca vio. Las etiquetas se usan para armar las
tareas y para medir; dónde las recibe la red está dicho en cada prueba.

  suma            (1a) suma con acarreo de números cuyos dígitos son imágenes MNIST,
                  columna por columna, con percepción entrenada aparte con etiquetas.
                  Referencias: el mismo Thinker con símbolos perfectos (oráculo) y la cota
                  si los errores de percepción fueran independientes:
                      exacta <= p_img^(2n) * exacta_oráculo(n)
  conjunta        (1b) percepción y razonamiento aprendiendo a la vez, desde cero, con el
                  dígito de cada columna como única señal: el Thinker recibe el objetivo y
                  la percepción el mismo acierto/fallo como recompensa global (R-STDP, el
                  único canal que ofrece la arquitectura). Escalera para ubicar el quiebre:
                      L0 identidad      una imagen -> su dígito
                      L1 a+b, b símbolo una imagen y un símbolo perfecto -> (a+b) mod 10
                      L2 a+b imágenes   dos imágenes -> (a+b) mod 10 (sin acarreo)
                      L3 suma completa  varias columnas con acarreo
                  La percepción se mide con la mejor biyección símbolo -> dígito (pureza),
                  porque al Thinker le sirve cualquier código consistente.
  ligadura        (2) en cada episodio se ligan 4 pares (imagen de clave, imagen de valor)
                  y se pregunta con OTRA imagen del dígito clave. Mitad de las preguntas
                  son claves no ligadas, que valen un dato permanente. Cota independiente:
                  ligadas <= p^3 * oráculo, no ligadas <= p * oráculo.
  modal           (3) asociación imagen <-> dígito hablado (FSDD) sin etiquetas: un área de
                  asociación ve pares que co-ocurren (imagen y voz del mismo dígito) y se
                  le pide completar una modalidad a partir de la otra con ejemplares no
                  vistos. Controles: memoria perfecta (cota del protocolo), pares sin
                  relación (azar), dos cortezas que se enseñan símbolos entre sí y
                  referencia con etiquetas.
  interferencia   (4a) ¿el refuerzo que viene del razonamiento degrada una percepción
                  entrenada con etiquetas? (4b) ¿los símbolos ruidosos de la percepción
                  degradan lo que el Thinker ya sabía con símbolos perfectos?
  resumen         junta los CSV por semilla en results/integracion_resumen.csv

  python experiments/exp_integration.py <prueba> [--seed S] [--rapida]

(--rapida: tamaños mínimos, sólo imprime; sirve de prueba de humo.)

Cada prueba con una semilla tarda menos de 10 minutos en un núcleo. Escribe
results/integracion_<prueba>_seed<S>.csv en formato largo: una fila por métrica.
"""
from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np

from common import RESULTS, ROOT, CsvLog
from bioelastic import AudioEncoder, BioElasticNet, Cortex
from bioelastic.datasets import load_fsdd, load_mnist
from bioelastic.thinker import Thinker
from exp_vision import CONFIGS, encoder

CACHE = ROOT / "ckpts_integracion"          # percepciones ya entrenadas (no va a git)
END = 20                                    # fin del número: la columna del último acarreo
HEADER = ["prueba", "variante", "semilla", "metrica", "valor", "referencia", "nota"]


# ====================================================================== utilidades
def pureza(conf: np.ndarray) -> float:
    """Exactitud con la mejor biyección símbolo -> dígito. `conf[dígito, símbolo]`."""
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-conf)
    return float(conf[r, c].sum() / max(conf.sum(), 1))


def cota_independiente(p: float, imagenes: int, oraculo: float) -> float:
    """Exactitud esperada si cada imagen se percibe bien con probabilidad p, de forma
    independiente, y el razonamiento acierta como con símbolos perfectos."""
    return float(p) ** int(imagenes) * float(oraculo)


def plasticidad(th, on: bool):
    """Congela o libera el aprendizaje del Thinker. Cerrar el episodio con reset() antes
    de congelar aplica su última actualización (así lo hace exp_thinking)."""
    th.reset()
    if hasattr(th, "plastic"):
        th.plastic = bool(on)


class Digitos:
    """Imágenes MNIST por dígito. Pools disjuntos:
        pre    train[:n_pre]  entrena la percepción con etiquetas
        train  train[n_pre:]  episodios de entrenamiento (imágenes que la percepción no vio)
        test   t10k           sólo para evaluar
    """

    def __init__(self, n_pre: int = 10000, arrays=None):
        Xtr, ytr, Xte, yte = load_mnist() if arrays is None else arrays
        self.n_pre = int(n_pre)
        self.X = dict(pre=Xtr[:n_pre], train=Xtr[n_pre:], test=Xte)
        self.y = dict(pre=ytr[:n_pre], train=ytr[n_pre:], test=yte)
        self.por = {s: [np.flatnonzero(y == d) for d in range(10)] for s, y in self.y.items()}

    def una(self, split: str, d: int, rng) -> np.ndarray:
        idx = self.por[split][int(d)]
        return self.X[split][idx[int(rng.integers(len(idx)))]]


class Vista:
    """Percepción visual: el Cortex de exp_vision (2 niveles: 98.5 % con MNIST completo).
    Su decisión (argmax o muestreo) es el símbolo que recibe el Thinker."""

    def __init__(self, seed: int, config: str = "depth2"):
        self.config = config
        self.net = Cortex((28, 28, 1), 10, seed=seed, **CONFIGS[config])
        self._E = encoder()

    def simbolo(self, img, explore: bool = False) -> int:
        return int(self.net.forward(self._E(img), explore=explore))

    def entrenar(self, X, y):
        for x, t in zip(X, y):
            self.net.step(self._E(x), int(t))

    def reforzar(self, img, sym: int, r: float):
        """Recompensa global sobre la decisión `sym` que se tomó ante `img`. Se vuelve a
        pasar la imagen porque el Cortex sólo guarda el estado del último pase."""
        self.net.forward(self._E(img))
        self.net.pred = int(sym)
        self.net.reinforce(float(r))

    def confusion(self, X, y) -> np.ndarray:
        conf = np.zeros((10, 10), dtype=np.int64)
        for x, t in zip(X, y):
            conf[int(t), self.simbolo(x)] += 1
        return conf


def vista_entrenada(datos: Digitos, seed: int, n: int, config: str = "depth2") -> Vista:
    """Percepción entrenada con etiquetas sobre train[:n], cacheada en disco por semilla."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"vista_{config}_n{n}_s{seed}.npz"
    v = Vista(seed, config)
    if path.exists():
        v.net = Cortex.load(path)
        return v
    v.entrenar(datos.X["pre"][:n], datos.y["pre"][:n])
    v.net.save(path)
    return v


def muestra_test(datos: Digitos, rng, n: int):
    idx = rng.choice(len(datos.y["test"]), size=min(n, len(datos.y["test"])), replace=False)
    return datos.X["test"][idx], datos.y["test"][idx]


def oraculo(img, d, explore):
    """Percepción perfecta: el símbolo es la etiqueta."""
    return int(d)


def con_vista(v: Vista):
    return lambda img, d, explore: v.simbolo(img, explore)


# ====================================================================== 1. suma con acarreo
def episodio_suma(th, ver, datos, split, rng, n, learn, reforzar=None):
    """Suma de dos números de n dígitos, columna por columna desde la menos significativa;
    al final, la columna END pide el último acarreo. `ver(img, dígito, explore)` devuelve
    el símbolo percibido. `reforzar([(img, símbolo), ...], r)` recibe el acierto de cada
    columna como recompensa (+1/-1)."""
    a = rng.integers(10, size=n); b = rng.integers(10, size=n)
    th.reset()
    carry = 0; cols = []; vistos = []
    for i in range(n + 1):
        if i < n:
            ia, ib = datos.una(split, a[i], rng), datos.una(split, b[i], rng)
            sa, sb = ver(ia, a[i], learn), ver(ib, b[i], learn)
            vistos += [sa == a[i], sb == b[i]]
            s = int(a[i]) + int(b[i]) + carry
            d, carry = s % 10, s // 10
            out = th.answer((sa, 10 + sb), explore=learn)
        else:
            d = carry
            out = th.answer(END, explore=learn)
        if learn:
            th.feedback(target=d)
        cols.append(out == d)
        if learn and reforzar is not None and i < n:
            reforzar([(ia, sa), (ib, sb)], 1.0 if out == d else -1.0)
    return dict(exacta=all(cols), columnas=cols, vistos=vistos)


def entrenar_suma(th, ver, datos, rng, episodios, reforzar=None, curva=None, cada=500):
    """Episodios de 1 a 4 dígitos. `curva` acumula (episodio, acierto por columna)."""
    win = []
    for e in range(episodios):
        r = episodio_suma(th, ver, datos, "train", rng, int(rng.integers(1, 5)), True, reforzar)
        win += r["columnas"]
        if curva is not None and (e + 1) % cada == 0:
            curva.append((e + 1, float(np.mean(win)))); win = []


def evaluar_suma(th, ver, datos, rng, n, episodios) -> dict:
    plasticidad(th, False)
    res = [episodio_suma(th, ver, datos, "test", rng, n, False) for _ in range(episodios)]
    plasticidad(th, True)
    ex = np.array([r["exacta"] for r in res])
    ok = np.array([all(r["vistos"]) for r in res])
    return dict(exacta=float(ex.mean()),
                columna=float(np.mean([c for r in res for c in r["columnas"][:-1]])),
                imagen=float(np.mean([v for r in res for v in r["vistos"]])) if res[0]["vistos"] else 1.0,
                todas_bien=float(ok.mean()),
                exacta_si_todas_bien=float(ex[ok].mean()) if ok.any() else float("nan"),
                exacta_si_alguna_mal=float(ex[~ok].mean()) if (~ok).any() else float("nan"))


def thinker_suma(seed, registers=2):
    return Thinker(21, registers=registers, seed=seed)


def prueba_suma(seed, tam, out):
    """1a: percepción entrenada aparte con etiquetas + Thinker."""
    rng = np.random.default_rng(1000 + seed)
    datos = Digitos(tam["n_pre"])
    ref = {}
    # Oráculo: símbolos perfectos. R=1 es la configuración de T3 (binaria) en exp_thinking.
    for R in (1, 2):
        th = thinker_suma(seed, R)
        entrenar_suma(th, oraculo, datos, rng, tam["ep_suma"])
        for n in (1, 3):
            m = evaluar_suma(th, oraculo, datos, rng, n, tam["ep_eval"])
            ref[(R, n)] = m["exacta"]
            out("suma", f"oraculo R={R}", f"exacta n={n}", m["exacta"], 10.0 ** -(n + 1), "azar")
            out("suma", f"oraculo R={R}", f"columna n={n}", m["columna"], 0.1, "azar")
        if R == 2:
            th_oraculo = th
    v = vista_entrenada(datos, seed, tam["n_pre"])
    Xt, yt = muestra_test(datos, rng, tam["n_test"])
    p = float(np.trace(v.confusion(Xt, yt)) / len(yt))
    out("suma", "vista", "exactitud imagen (test)", p, "", f"depth2, {tam['n_pre']} etiquetas")
    variantes = [("1a tubería (Thinker entrenado con oráculo)", th_oraculo)]
    th = thinker_suma(seed, 2)
    entrenar_suma(th, con_vista(v), datos, rng, tam["ep_suma"])
    variantes.append(("1a ruidosa (Thinker entrenado sobre lo percibido)", th))
    for name, th in variantes:
        for n in (1, 3):
            m = evaluar_suma(th, con_vista(v), datos, rng, n, tam["ep_eval"])
            cota = cota_independiente(p, 2 * n, ref[(2, n)])
            out("suma", name, f"exacta n={n}", m["exacta"], cota, "cota indep. p^2n*oráculo")
            out("suma", name, f"columna n={n}", m["columna"], "", "")
            out("suma", name, f"todas las imágenes bien n={n}", m["todas_bien"], p ** (2 * n), "p^2n")
            out("suma", name, f"exacta | todas bien n={n}", m["exacta_si_todas_bien"], ref[(2, n)], "oráculo")
            out("suma", name, f"exacta | alguna mal n={n}", m["exacta_si_alguna_mal"], 0.0, "")


# ====================================================================== 1b. aprendizaje conjunto
def episodio_escalera(nivel, th, ver, datos, split, rng, learn, reforzar=None):
    """Una columna de los niveles L0-L2. Devuelve (acierto, [(img, símbolo, dígito)])."""
    a, b = int(rng.integers(10)), int(rng.integers(10))
    ia = datos.una(split, a, rng)
    sa = ver(ia, a, learn)
    vistos = [(ia, sa, a)]
    th.reset()
    if nivel == "L0":
        y = a; tok = (sa,)
    elif nivel == "L1":
        y = (a + b) % 10; tok = (sa, 10 + b)
    else:
        ib = datos.una(split, b, rng)
        sb = ver(ib, b, learn)
        vistos.append((ib, sb, b))
        y = (a + b) % 10; tok = (sa, 10 + sb)
    out = th.answer(tok, explore=learn)
    if learn:
        th.feedback(target=y)
        if reforzar is not None:
            reforzar([(i, s) for i, s, _ in vistos], 1.0 if out == y else -1.0)
    return out == y, vistos


def prueba_conjunta(seed, tam, out):
    """1b: percepción desde cero + Thinker desde cero; única señal: el dígito de la columna."""
    rng = np.random.default_rng(2000 + seed)
    datos = Digitos(tam["n_pre"])
    Xt, yt = muestra_test(datos, rng, tam["n_test_pureza"])
    for nivel in tam["niveles"]:
        v = Vista(seed)
        refuerzo = lambda pares, r: [v.reforzar(img, s, r) for img, s in pares]
        th = Thinker(10 if nivel == "L0" else (20 if nivel in ("L1", "L2") else 21),
                     registers=2 if nivel == "L3" else 0, seed=seed)
        t0 = time.time(); win = []
        for e in range(tam["ep_conjunta"]):
            if nivel == "L3":
                r = episodio_suma(th, con_vista(v), datos, "train", rng, int(rng.integers(1, 5)), True, refuerzo)
                win += r["columnas"]
            else:
                win.append(episodio_escalera(nivel, th, con_vista(v), datos, "train", rng, True, refuerzo)[0])
            if (e + 1) % tam["cada"] == 0:
                conf = v.confusion(Xt[:500], yt[:500])
                out("conjunta", nivel, f"curva columna ep={e + 1}", float(np.mean(win)), 0.1, "azar")
                out("conjunta", nivel, f"curva pureza ep={e + 1}", pureza(conf), 0.1,
                    f"símbolos usados={int((conf.sum(0) > 0).sum())}")
                win = []
        conf = v.confusion(Xt, yt)
        pz = pureza(conf)
        out("conjunta", nivel, "pureza percepción (test)", pz, 0.1, "azar")
        out("conjunta", nivel, "exactitud identidad (test)", float(np.trace(conf) / conf.sum()), 0.1, "azar")
        out("conjunta", nivel, "símbolos distintos usados", int((conf.sum(0) > 0).sum()), 10, "")
        plasticidad(th, False)
        if nivel == "L3":
            m = evaluar_suma(th, con_vista(v), datos, rng, 3, tam["ep_eval"])
            out("conjunta", nivel, "exacta n=3 (test)", m["exacta"], 1e-4, "azar")
            out("conjunta", nivel, "columna n=3 (test)", m["columna"], 0.1, "azar")
            out("conjunta", nivel, "exacta n=3: cota con la pureza medida", "",
                cota_independiente(pz, 6, 1.0), "pureza^6 (Thinker perfecto)")
        else:
            acc = float(np.mean([episodio_escalera(nivel, th, con_vista(v), datos, "test", rng, False)[0]
                                 for _ in range(tam["ep_eval"])]))
            k = 1 if nivel in ("L0", "L1") else 2
            out("conjunta", nivel, "acierto tarea (test)", acc, cota_independiente(pz, k, 1.0),
                f"pureza^{k} (cota si el Thinker fuera perfecto)")
        out("conjunta", nivel, "segundos", round(time.time() - t0, 1), "", "")


# ====================================================================== 2. ligadura perceptual
def episodio_ligadura(th, ver, datos, split, rng, learn, ligada=None, F=None):
    """4 pares (clave, valor) de imágenes; la clave son dígitos distintos y el valor llega
    con el símbolo desplazado en 10 (rol de valor). Pregunta con OTRA imagen del dígito:
    si está ligado responde el valor; si no, el dato permanente F[clave] (+10)."""
    claves = rng.permutation(10)[:5]
    valores = rng.integers(10, size=4)
    th.reset()
    ok_perc = True
    for k, val in zip(claves[:4], valores):
        sk = ver(datos.una(split, k, rng), k, learn)
        sv = ver(datos.una(split, val, rng), val, learn)
        th.observe(sk); th.observe(10 + sv); th.observe(None)
        ok_perc &= (sk == k) and (sv == val)
    ligada = rng.random() < 0.5 if ligada is None else ligada
    j = int(rng.integers(4))
    q = int(claves[j]) if ligada else int(claves[4])
    y = 10 + (int(valores[j]) if ligada else int(F[q]))
    sq = ver(datos.una(split, q, rng), q, learn)
    ok_perc &= sq == q
    out = th.answer(sq, explore=learn)
    if learn:
        th.feedback(target=y)
    return out == y, bool(ok_perc)


def evaluar_ligadura(th, ver, datos, rng, F, episodios):
    plasticidad(th, False)
    res = {}
    for lig in (True, False):
        r = [episodio_ligadura(th, ver, datos, "test", rng, False, lig, F) for _ in range(episodios)]
        hit = np.array([x[0] for x in r]); ok = np.array([x[1] for x in r])
        res[lig] = dict(acierto=float(hit.mean()), todas_bien=float(ok.mean()),
                        si_bien=float(hit[ok].mean()) if ok.any() else float("nan"),
                        si_mal=float(hit[~ok].mean()) if (~ok).any() else float("nan"))
    plasticidad(th, True)
    return res


def prueba_ligadura(seed, tam, out):
    rng = np.random.default_rng(3000 + seed)
    F = np.random.default_rng(7).permutation(10)               # dato permanente de cada clave
    datos = Digitos(tam["n_pre"])
    th_o = Thinker(20, memory=True, seed=seed)
    for _ in range(tam["ep_ligadura"]):
        episodio_ligadura(th_o, oraculo, datos, "train", rng, True, None, F)
    ref = evaluar_ligadura(th_o, oraculo, datos, rng, F, tam["ep_eval"])
    for lig, name in ((True, "ligadas"), (False, "no ligadas")):
        out("ligadura", "oraculo", f"acierto {name}", ref[lig]["acierto"], 0.1, "azar")
    Xt, yt = muestra_test(datos, rng, tam["n_test"])
    vistas = [("vista fuerte", vista_entrenada(datos, seed, tam["n_pre"])),
              ("vista débil", vista_entrenada(datos, seed, tam["n_debil"], "depth1"))]
    for vname, v in vistas:
        p = float(np.trace(v.confusion(Xt, yt)) / len(yt))
        out("ligadura", vname, "exactitud imagen (test)", p, "", v.config)
        th = Thinker(20, memory=True, seed=seed)
        for _ in range(tam["ep_ligadura"]):
            episodio_ligadura(th, con_vista(v), datos, "train", rng, True, None, F)
        for tname, t in (("tubería", th_o), ("ruidosa", th)):
            res = evaluar_ligadura(t, con_vista(v), datos, rng, F, tam["ep_eval"])
            for lig, name, k in ((True, "ligadas", 3), (False, "no ligadas", 1)):
                m = res[lig]
                var = f"{vname} {tname}"
                out("ligadura", var, f"acierto {name}", m["acierto"],
                    cota_independiente(p, k, ref[lig]["acierto"]), f"p^{k}*oráculo")
                out("ligadura", var, f"acierto {name} | 9 imágenes bien", m["si_bien"], ref[lig]["acierto"], "oráculo")
                out("ligadura", var, f"acierto {name} | alguna mal", m["si_mal"], "", "")


# ====================================================================== 3. asociación entre modalidades
def _unidad(x):
    x = np.asarray(x, dtype=np.float32).ravel()
    return x / max(float(np.sqrt(x @ x)), 1e-8)


def conjunto(img, aud):
    """Volea de las dos modalidades con el mismo peso (cada mitad con norma 1), en [0, 1]."""
    v = np.concatenate([_unidad(img), _unidad(aud)])
    return v / max(float(v.max()), 1e-8)


def completar(mem: BioElasticNet, cue: np.ndarray, lado: str, d_img: int) -> np.ndarray:
    """Completa un patrón desde una sola modalidad: la otra mitad de la volea va en
    silencio; la neurona de la memoria que dispara primero (o la de mayor potencial si la
    columna queda en silencio) devuelve la mitad que guardó de la otra modalidad."""
    z = np.zeros(mem.N, dtype=np.float32)
    if lado == "img":
        z[:d_img] = _unidad(cue)
    else:
        z[d_img:] = _unidad(cue)
    z /= max(float(z.max()), 1e-8)
    idx = np.flatnonzero(z > 0)
    v = mem.layer.forward(idx, z[idx])
    if len(v.winners) == 0:
        return np.zeros(mem.N - d_img if lado == "img" else d_img, dtype=np.float32)
    w = mem.layer.dense(int(v.winners[0]))
    return w[d_img:] if lado == "img" else w[:d_img]


def p_at_1(recuperados, etiquetas_q, banco, etiquetas_b) -> float:
    """¿El vecino más cercano (coseno) del patrón recuperado, en el banco de ejemplares de
    test de la otra modalidad, es del mismo dígito que la consulta? Azar: 10 %."""
    B = np.stack([_unidad(b) for b in banco])
    R = np.stack([_unidad(r) for r in recuperados])
    nn = np.argmax(R @ B.T, axis=1)
    return float(np.mean(etiquetas_b[nn] == etiquetas_q))


def prueba_modal(seed, tam, out):
    rng = np.random.default_rng(4000 + seed)
    datos = Digitos(tam["n_pre"])
    recs, sr = load_fsdd()
    enc_a = AudioEncoder(sr=sr, n_bands=32, n_frames=24, dyn_range_db=35)
    E = encoder()
    a_tr = [(enc_a(r[0]).ravel(), r[1]) for r in recs if r[3] >= 5]
    a_te = [(enc_a(r[0]).ravel(), r[1]) for r in recs if r[3] < 5]
    a_por = [[x for x, d in a_tr if d == k] for k in range(10)]
    Ate = np.stack([x for x, _ in a_te]); yAte = np.array([d for _, d in a_te])
    qi = rng.choice(len(datos.y["test"]), size=tam["n_q_img"], replace=False)
    Ite = np.stack([E(x).ravel() for x in datos.X["test"][qi]]); yIte = datos.y["test"][qi]
    D_IMG = 28 * 28

    def pares(n, relacion=True):
        for _ in range(n):
            d = int(rng.integers(10))
            img = E(datos.una("train", d, rng)).ravel()
            da = d if relacion else int(rng.integers(10))
            aud = a_por[da][int(rng.integers(len(a_por[da])))]
            yield img, aud

    def medir(nombre, mem):
        rec_a = [completar(mem, x, "img", D_IMG) for x in Ite]
        out("modal", nombre, "imagen -> voz P@1", p_at_1(rec_a, yIte, Ate, yAte), 0.1, "azar")
        rec_i = [completar(mem, x, "aud", D_IMG) for x in Ate]
        out("modal", nombre, "voz -> imagen P@1", p_at_1(rec_i, yAte, Ite, yIte), 0.1, "azar")
        out("modal", nombre, "neuronas", mem.layer.n, "", f"{tam['n_pares']} pares")

    for nombre, relacion in (("memoria asociativa", True), ("control: pares sin relación", False)):
        mem = BioElasticNet(D_IMG + Ate.shape[1], 1, seed=seed,
                            layer=dict(fan_in=D_IMG + Ate.shape[1], theta=tam["theta_modal"]))
        t0 = time.time()
        for img, aud in pares(tam["n_pares"], relacion):
            mem.observe(conjunto(img, aud))
        medir(nombre, mem)
        out("modal", nombre, "segundos", round(time.time() - t0, 1), "", "")
    # Cota del protocolo: lo recuperado es un ejemplar de entrenamiento del dígito correcto.
    rec_a = [a_por[d][int(rng.integers(len(a_por[d])))] for d in yIte]
    out("modal", "control: memoria perfecta", "imagen -> voz P@1", p_at_1(rec_a, yIte, Ate, yAte), "", "cota del protocolo")
    rec_i = [E(datos.una("train", d, rng)).ravel() for d in yAte]
    out("modal", "control: memoria perfecta", "voz -> imagen P@1", p_at_1(rec_i, yAte, Ite, yIte), "", "cota del protocolo")

    # Dos cortezas que se enseñan símbolos entre sí (sin etiquetas): cada una aprende a
    # predecir el símbolo que eligió la otra ante el mismo dígito.
    SHEET = (24, 32, 1)
    ci = Cortex((28, 28, 1), 10, seed=seed, **CONFIGS["depth2"])
    ca = Cortex(SHEET, 10, seed=seed + 50, **CONFIGS["depth2"])
    for img, aud in pares(tam["n_pares_mutuos"]):
        si = ci.forward(img.reshape(28, 28, 1), explore=True)
        sa = ca.forward(aud.reshape(SHEET), explore=True)
        ci.teach(sa); ca.teach(si)
    _cruce(out, "símbolos mutuos (sin etiquetas)", ci, ca, Ite, yIte, Ate, yAte, SHEET)
    # Referencia con etiquetas: cada corteza aprende su modalidad con el dígito.
    ci = Cortex((28, 28, 1), 10, seed=seed, **CONFIGS["depth2"])
    ca = Cortex(SHEET, 10, seed=seed + 50, **CONFIGS["depth2"])
    for x, d in zip(datos.X["pre"][:tam["n_ref"]], datos.y["pre"][:tam["n_ref"]]):
        ci.step(E(x), int(d))
    for i in rng.permutation(len(a_tr)):
        ca.step(a_tr[i][0].reshape(SHEET), int(a_tr[i][1]))
    _cruce(out, "referencia con etiquetas", ci, ca, Ite, yIte, Ate, yAte, SHEET)


def _cruce(out, nombre, ci, ca, Ite, yIte, Ate, yAte, SHEET):
    """Recuperación por símbolo compartido: ante una consulta se elige al azar un ejemplar
    de la otra modalidad al que su corteza le asigna el mismo símbolo."""
    si = np.array([ci.predict(x.reshape(28, 28, 1)) for x in Ite])
    sa = np.array([ca.predict(x.reshape(SHEET)) for x in Ate])
    rng = np.random.default_rng(0)

    def recup(s_q, y_q, s_b, y_b):
        hits = []
        for s, y in zip(s_q, y_q):
            m = np.flatnonzero(s_b == s)
            hits.append(len(m) > 0 and y_b[m[int(rng.integers(len(m)))]] == y)
        return float(np.mean(hits))

    out("modal", nombre, "imagen -> voz P@1", recup(si, yIte, sa, yAte), 0.1, "azar")
    out("modal", nombre, "voz -> imagen P@1", recup(sa, yAte, si, yIte), 0.1, "azar")
    ci_conf = np.zeros((10, 10), dtype=np.int64); np.add.at(ci_conf, (yIte, si), 1)
    ca_conf = np.zeros((10, 10), dtype=np.int64); np.add.at(ca_conf, (yAte, sa), 1)
    out("modal", nombre, "pureza imagen", pureza(ci_conf), 0.1, f"símbolos usados={int((ci_conf.sum(0) > 0).sum())}")
    out("modal", nombre, "pureza voz", pureza(ca_conf), 0.1, f"símbolos usados={int((ca_conf.sum(0) > 0).sum())}")


# ====================================================================== 4. interferencia
def prueba_interferencia(seed, tam, out):
    rng = np.random.default_rng(5000 + seed)
    datos = Digitos(tam["n_pre"])
    Xt, yt = muestra_test(datos, rng, tam["n_test"])
    base = vista_entrenada(datos, seed, tam["n_pre"])
    p0 = float(np.trace(base.confusion(Xt, yt)) / len(yt))
    out("interferencia", "4a", "vista antes (test)", p0, "", "entrenada con etiquetas")
    th0 = thinker_suma(seed, 2)
    entrenar_suma(th0, oraculo, datos, rng, tam["ep_suma"])
    T0 = evaluar_suma(th0, oraculo, datos, rng, 3, tam["ep_eval"])["exacta"]
    out("interferencia", "4b", "Thinker antes: exacta n=3 con oráculo", T0, "", "")

    # 4a. La percepción entrenada recibe la recompensa del razonamiento.
    for name, th_ini in (("Thinker pre-entrenado", th0), ("Thinker desde cero", None)):
        for plastica in (True, False):
            v = copy.deepcopy(base)
            th = copy.deepcopy(th0) if th_ini is not None else thinker_suma(seed, 2)
            ref = (lambda pares, r: [v.reforzar(img, s, r) for img, s in pares]) if plastica else None
            entrenar_suma(th, con_vista(v), datos, rng, tam["ep_interf"], ref)
            conf = v.confusion(Xt, yt)
            var = f"4a {name}, vista {'con refuerzo' if plastica else 'congelada (control)'}"
            out("interferencia", var, "vista después (test)", float(np.trace(conf) / conf.sum()), p0, "antes")
            out("interferencia", var, "pureza vista después", pureza(conf), p0, "antes")
            m = evaluar_suma(th, con_vista(v), datos, rng, 3, tam["ep_eval"])
            out("interferencia", var, "suma exacta n=3 (test)", m["exacta"], "", "")

    # 4b. El Thinker que sabía sumar con símbolos perfectos aprende sobre lo percibido.
    debil = vista_entrenada(datos, seed, tam["n_debil"], "depth1")
    pd = float(np.trace(debil.confusion(Xt, yt)) / len(yt))
    out("interferencia", "4b", "vista débil (test)", pd, "", "depth1")
    for name, ver in (("control: sigue con oráculo", oraculo), ("vista fuerte", con_vista(base)),
                      ("vista débil", con_vista(debil))):
        th = copy.deepcopy(th0)
        entrenar_suma(th, ver, datos, rng, tam["ep_interf"])
        m = evaluar_suma(th, oraculo, datos, rng, 3, tam["ep_eval"])
        out("interferencia", f"4b {name}", "Thinker después: exacta n=3 con oráculo", m["exacta"], T0, "antes")
        out("interferencia", f"4b {name}", "Thinker después: columna n=3 con oráculo", m["columna"], "", "")


# ====================================================================== principal
TAM = dict(
    completa=dict(n_pre=10000, n_debil=1000, n_test=5000, n_test_pureza=2000, ep_suma=6000, ep_eval=500,
                  ep_conjunta=4000, cada=1000, niveles=("L0", "L1", "L2", "L3"), ep_ligadura=3000,
                  n_q_img=1000, n_pares=3000, n_pares_mutuos=3000, n_ref=5000, theta_modal=0.7,
                  ep_interf=3000),
    rapida=dict(n_pre=300, n_debil=200, n_test=200, n_test_pureza=100, ep_suma=60, ep_eval=20,
                ep_conjunta=20, cada=10, niveles=("L0", "L3"), ep_ligadura=40,
                n_q_img=50, n_pares=60, n_pares_mutuos=20, n_ref=50, theta_modal=0.7,
                ep_interf=20),
)
PRUEBAS = dict(suma=prueba_suma, conjunta=prueba_conjunta, ligadura=prueba_ligadura,
               modal=prueba_modal, interferencia=prueba_interferencia)


def resumen():
    """Media, mínimo y máximo por (prueba, variante, métrica) sobre las semillas."""
    filas = {}
    for p in sorted(RESULTS.glob("integracion_*_seed*.csv")):
        with p.open() as f:
            for r in csv.DictReader(f):
                try:
                    v = float(r["valor"])
                except ValueError:
                    continue
                k = (r["prueba"], r["variante"], r["metrica"])
                filas.setdefault(k, dict(v=[], ref=[], nota=r["nota"]))
                filas[k]["v"].append(v)
                try:
                    filas[k]["ref"].append(float(r["referencia"]))
                except ValueError:
                    pass
    log = CsvLog("integracion_resumen", ["prueba", "variante", "metrica", "media", "min", "max",
                                         "semillas", "referencia_media", "nota"])
    for (pr, var, met), d in filas.items():
        v = np.array(d["v"])
        log.row(pr, var, met, float(v.mean()), float(v.min()), float(v.max()), len(v),
                float(np.mean(d["ref"])) if d["ref"] else "", d["nota"])
    log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prueba", choices=list(PRUEBAS) + ["resumen"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rapida", action="store_true", help="tamaños mínimos (humo)")
    args = ap.parse_args()
    if args.prueba == "resumen":
        resumen()
        return
    t0 = time.time()
    if args.rapida:                                       # humo: sólo imprime, no escribe CSV
        PRUEBAS[args.prueba](args.seed, TAM["rapida"], lambda *r: print(*r, sep="  ", flush=True))
        print("segundos", round(time.time() - t0, 1))
        return
    log = CsvLog(f"integracion_{args.prueba}_seed{args.seed}", HEADER)
    PRUEBAS[args.prueba](args.seed, TAM["completa"], lambda *r: log.row(r[0], r[1], args.seed, *r[2:]))
    log.row(args.prueba, "total", args.seed, "segundos", round(time.time() - t0, 1), "", "")
    log.close()


if __name__ == "__main__":
    main()
