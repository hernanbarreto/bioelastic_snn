"""Integración, ronda 2: que percepción y razonamiento aprendan juntos.

La ronda 1 (exp_integration.py) mostró dónde se rompe la tubería percepción -> Thinker:
el Thinker que aprende sobre lo percibido queda debajo de la cota, aprender todo desde
cero con el dígito de la suma como única señal no funciona (la recompensa global no
reparte el crédito entre dos imágenes y la regla), y los símbolos ruidosos degradan lo
que el Thinker ya sabía. Acá se prueban dos canales nuevos entre las dos partes:

  crédito simbólico   ante un error, el Thinker busca (con su conocimiento establecido y
  (`cred`)            sin efectos) qué símbolo en cada ranura de la pregunta habría dado
                      el objetivo; si hay una sola explicación (contando también la de un
                      registro mal escrito), se la enseña a la percepción como objetivo
                      local y el Thinker no aprende de ese error. `cred` considera todos
                      los símbolos; `cred2`, sólo la segunda opción de la percepción.
  confianza           la percepción pasa con qué certeza eligió cada símbolo; el Thinker
  (`conf`, `confcos`) no aprende de una respuesta que se apoyó en algo dudoso (pregunta,
                      recuerdo o registros). Certeza: la probabilidad de la decisión
                      (`conf`) o esa probabilidad por el parecido de la imagen al
                      prototipo ganador a la cuarta (`confcos`). Umbral: el cuantil
                      `duda` (20 %) de la certeza sobre imágenes sin etiqueta del pool de
                      entrenamiento: la percepción duda de lo menos seguro que suele ver.

Pruebas (cada corrida, una semilla, < 10 minutos en un núcleo):
  acarreo     suma decimal con símbolos perfectos: cuántos registros hacen falta para
              que el acarreo salga en todas las semillas (registros que empiezan en 0,
              como antes, o escritos por el propio escritor al inicio: `reg_reset`).
  suma        1a: percepción entrenada aparte con etiquetas y Thinker que aprende sobre
              lo percibido. Variantes base / conf / cred2 / conf+cred2. Además, pruebas
              para romperlo sobre lo ya entrenado: ruido gaussiano en las imágenes,
              dígitos tipográficos (un estilo que la percepción nunca vio) y números más
              largos que en el entrenamiento.
  escalera    L0-L2 desde cero (ver exp_integration.py): base / H / credU+H / cred2+H.
  conjunta    L3 desde cero (suma completa): base / H / credU+H / cred2+H.
              (H: homeostasis de las salidas de la percepción; credU: todas las
              alternativas y basta con una sola explicación, aunque otras no se conozcan.)
  ligadura    vista débil, Thinker aprendiendo sobre lo percibido: base / conf /
              confcos / cred / cred2 / confcos+cred2.
  interf      4b: el Thinker que sabía sumar sigue aprendiendo sobre la vista débil;
              se lo vuelve a medir con símbolos perfectos. Control (sigue con oráculo) /
              base / conf / confcos / cred2 / confcos+cred2.
  estilo      adaptación sin etiquetas a dígitos tipográficos: la percepción (MNIST)
              recibe sólo el crédito del Thinker que ya sabe sumar. Control (nadie
              aprende) / refuerzo +-1 de la ronda 1 / cred2 / cred.
  resumen     results/integracion2_resumen.csv

  python experiments/exp_integration2.py <prueba> [--seed S] [--variante V] [--rapida]

Toda evaluación usa ejemplares del split oficial de test (MNIST t10k). Escribe
results/integracion2_<prueba>[_<variante>]_seed<S>.csv (formato largo, una fila por
métrica). Ninguna cifra de acá está validada: es lo medido.
"""
from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np

import exp_integration as ei
from common import RESULTS, CsvLog
from bioelastic.thinker import Thinker

HEADER = ei.HEADER
END = ei.END
FUENTES = ("dejavu/DejaVuSans.ttf", "dejavu/DejaVuSans-Bold.ttf", "dejavu/DejaVuSerif.ttf",
           "dejavu/DejaVuSerif-Italic.ttf", "dejavu/DejaVuSansMono.ttf",
           "dejavu/DejaVuSansCondensed-Bold.ttf", "freefont/FreeSans.ttf",
           "freefont/FreeSerifBold.ttf", "freefont/FreeMono.ttf", "freefont/FreeSansOblique.ttf")


# ====================================================================== interfaz
class Percepcion:
    """Interfaz percepción -> Thinker de esta ronda: además del símbolo pasa con qué
    certeza lo eligió y qué otros símbolos consideraba (sus `k` mejores), y puede recibir
    de vuelta el símbolo que debió percibir (`ensenar`).

    certeza "p": probabilidad de la decisión en la competencia entre salidas (el softmax
    con el que el Cortex explora). "pcos": p por el coseno entre la imagen y el prototipo
    ganador del área de asociación, a la cuarta (cuán conocida le resulta la imagen)."""

    def __init__(self, vista, k: int = 10, certeza: str = "p"):
        self.v = vista
        self.k = int(k)
        self.certeza = certeza
        self.ensenadas = 0

    def ver(self, img, d, explore):
        s = self.v.simbolo(img, explore)
        p = self.v.net._p()
        c = float(p[s])
        if self.certeza == "pcos":
            cos = self.v.net.heads[0]._last.v.cos
            c *= float(cos[0]) ** 4 if len(cos) else 0.0
        alts = [int(x) for x in np.argsort(-p)[:self.k] if x != s][:self.k - 1]
        return s, c, alts

    def ensenar(self, img, sym: int):
        self.v.net.forward(self.v._E(img))
        self.v.net.teach(int(sym))
        self.ensenadas += 1

    def reforzar(self, img, sym, r):
        self.v.reforzar(img, sym, r)

    def umbral(self, imgs, q: float) -> float:
        """Cuantil `q` de la certeza sobre imágenes sin etiqueta (no aprende de ellas)."""
        return float(np.quantile([self.ver(x, -1, False)[1] for x in imgs], q))


class Oraculo:
    """Percepción perfecta y segura: el símbolo es la etiqueta."""
    k = 1

    def ver(self, img, d, explore):
        return int(d), 1.0, []


def _desplazar(alts, off):
    return [int(a) + off for a in alts]


def _credito(th, per, imgs, offs, digitos, st):
    """Si el Thinker explicó el error por una sola ranura, la percepción aprende ese
    símbolo para esa imagen. `st` cuenta créditos y si coinciden con el dígito real
    (sólo tiene sentido como precisión si el código de la percepción es la identidad)."""
    if not th.input_credit or not hasattr(per, "ensenar"):
        return
    slot, sym = th.input_credit[0]
    s = int(sym) - offs[slot]
    per.ensenar(imgs[slot], s)
    st["creditos"] = st.get("creditos", 0) + 1
    st["creditos_bien"] = st.get("creditos_bien", 0) + int(s == int(digitos[slot]))
    st.setdefault("pares", []).append((int(digitos[slot]), s))


# ====================================================================== suma
def episodio_suma(th, per, datos, split, rng, n, learn, cfg, st, transformar=None):
    """Suma de dos números de n dígitos columna por columna (como en la ronda 1) con la
    interfaz nueva. cfg: conf (pasar la certeza), cred (pasar alternativas y aplicar el
    crédito a la percepción), refuerzo (la percepción recibe +-1 por columna, como R-STDP)."""
    a = rng.integers(10, size=n); b = rng.integers(10, size=n)
    th.reset()
    carry = 0; cols = []; vistos = []
    for i in range(n + 1):
        alts = None
        if i < n:
            ia, ib = datos.una(split, a[i], rng), datos.una(split, b[i], rng)
            if transformar is not None:
                ia, ib = transformar(ia), transformar(ib)
            sa, ca, aa = per.ver(ia, a[i], learn)
            sb, cb, ab = per.ver(ib, b[i], learn)
            vistos += [sa == a[i], sb == b[i]]
            s = int(a[i]) + int(b[i]) + carry
            d, carry = s % 10, s // 10
            cert = min(ca, cb) if cfg.get("conf") else 1.0
            out = th.answer((sa, 10 + sb), explore=learn, certainty=cert)
            if cfg.get("cred"):
                alts = [aa, _desplazar(ab, 10)]
        else:
            d = carry
            out = th.answer(END, explore=learn)
        if learn:
            th.feedback(target=d, alternatives=alts)
            if i < n:
                _credito(th, per, (ia, ib), (0, 10), (a[i], b[i]), st)
                if cfg.get("refuerzo"):
                    r = 1.0 if out == d else -1.0
                    per.reforzar(ia, sa, r); per.reforzar(ib, sb, r)
        cols.append(out == d)
    return dict(exacta=all(cols), columnas=cols, vistos=vistos)


def entrenar_suma(th, per, datos, rng, episodios, cfg, st, curva=None, cada=1000):
    win = []
    for e in range(episodios):
        r = episodio_suma(th, per, datos, "train", rng, int(rng.integers(1, 5)), True, cfg, st)
        win += r["columnas"]
        if curva is not None and (e + 1) % cada == 0:
            curva.append((e + 1, float(np.mean(win)))); win = []


def evaluar_suma(th, per, datos, rng, n, episodios, transformar=None) -> dict:
    ei.plasticidad(th, False)
    st: dict = {}
    res = [episodio_suma(th, per, datos, "test", rng, n, False, {}, st, transformar)
           for _ in range(episodios)]
    ei.plasticidad(th, True)
    ex = np.array([r["exacta"] for r in res])
    ok = np.array([all(r["vistos"]) for r in res])
    return dict(exacta=float(ex.mean()),
                columna=float(np.mean([c for r in res for c in r["columnas"][:-1]])),
                imagen=float(np.mean([v for r in res for v in r["vistos"]])) if res[0]["vistos"] else 1.0,
                todas_bien=float(ok.mean()),
                exacta_si_todas_bien=float(ex[ok].mean()) if ok.any() else float("nan"),
                exacta_si_alguna_mal=float(ex[~ok].mean()) if (~ok).any() else float("nan"))


def thinker_suma(seed, R=2, reset="cero", doubt=0.0):
    return Thinker(21, registers=R, seed=seed, reg_reset=reset, doubt=doubt)


# ---------------------------------------------------------------- estilo no visto
def digitos_tipograficos(n_por_digito: int, seed: int):
    """Dígitos impresos con 10 tipografías (un estilo que la percepción, entrenada con
    MNIST manuscrito, nunca vio), con la normalización de MNIST: el dígito entra en una
    caja de 20x20 y se centra por su centro de masa en 28x28. Rotación de +-12 grados y
    tamaño variable. Devuelve (X uint8, y) o None si no hay tipografías instaladas."""
    from PIL import Image, ImageDraw, ImageFont
    from scipy import ndimage
    raiz = Path("/usr/share/fonts/truetype")
    fuentes = [raiz / f for f in FUENTES if (raiz / f).exists()]
    if not fuentes:
        return None
    rng = np.random.default_rng(seed)
    X, y = [], []
    for d in range(10):
        for _ in range(n_por_digito):
            f = ImageFont.truetype(str(fuentes[int(rng.integers(len(fuentes)))]), 64)
            img = Image.new("L", (96, 96), 0)
            ImageDraw.Draw(img).text((48, 48), str(d), fill=255, font=f, anchor="mm")
            img = img.rotate(float(rng.uniform(-12, 12)), resample=Image.BILINEAR)
            a = np.asarray(img)
            r, c = np.flatnonzero(a.max(1) > 0), np.flatnonzero(a.max(0) > 0)
            a = a[r[0]:r[-1] + 1, c[0]:c[-1] + 1]
            esc = float(rng.uniform(16, 20)) / max(a.shape)
            h, w = max(1, round(a.shape[0] * esc)), max(1, round(a.shape[1] * esc))
            a = np.asarray(Image.fromarray(a).resize((w, h), Image.LANCZOS), dtype=np.float32)
            out = np.zeros((28, 28), dtype=np.float32)
            cy, cx = ndimage.center_of_mass(a)
            y0, x0 = int(round(14 - cy)), int(round(14 - cx))
            y0, x0 = min(max(y0, 0), 28 - h), min(max(x0, 0), 28 - w)
            out[y0:y0 + h, x0:x0 + w] = a
            X.append(np.clip(out, 0, 255).astype(np.uint8)); y.append(d)
    return np.array(X), np.array(y, dtype=np.uint8)


class _Pool:
    """Pool de imágenes por dígito con la misma interfaz que `Digitos.una`."""

    def __init__(self, X, y):
        self.X, self.y = X, y
        self.por = [np.flatnonzero(y == d) for d in range(10)]

    def una(self, split, d, rng):
        idx = self.por[int(d)]
        return self.X[idx[int(rng.integers(len(idx)))]]


def ruido(sigma, rng):
    return lambda img: np.clip(img.astype(np.float32) + rng.normal(0, sigma * 255, img.shape),
                               0, 255).astype(np.uint8)


VARIANTES_SUMA = {"base": {}, "conf": dict(conf=True), "cred2": dict(cred=True),
                  "conf+cred2": dict(conf=True, cred=True)}


def prueba_suma(seed, tam, out, variante):
    """1a con la interfaz nueva y pruebas para romperlo."""
    cfg = VARIANTES_SUMA[variante]
    rng = np.random.default_rng(1000 + seed)
    datos = ei.Digitos(tam["n_pre"])
    th_o = thinker_suma(seed)
    entrenar_suma(th_o, Oraculo(), datos, rng, tam["ep_suma"], {}, {})
    ref = {n: evaluar_suma(th_o, Oraculo(), datos, rng, n, tam["ep_eval"])["exacta"] for n in tam["n_largos"]}
    for n in tam["n_largos"]:
        out("suma", "oraculo R=2", f"exacta n={n}", ref[n], "", "símbolos perfectos")
    v = ei.vista_entrenada(datos, seed, tam["n_pre"])
    per = Percepcion(v, k=2)
    Xt, yt = ei.muestra_test(datos, rng, tam["n_test"])
    p0 = float(np.trace(v.confusion(Xt, yt)) / len(yt))
    out("suma", variante, "vista antes (test)", p0, "", "depth2")
    cal = datos.X["train"][rng.choice(len(datos.y["train"]), tam["n_cal"], replace=False)]
    tau = per.umbral(cal, tam["duda"]) if cfg.get("conf") else 0.0
    th = thinker_suma(seed, doubt=tau)
    st: dict = {}
    t0 = time.time()
    entrenar_suma(th, per, datos, rng, tam["ep_suma"], cfg, st)
    out("suma", variante, "segundos entrenamiento", round(time.time() - t0, 1), "", "")
    p = float(np.trace(v.confusion(Xt, yt)) / len(yt))
    out("suma", variante, "vista después (test)", p, p0, "antes")
    out("suma", variante, "umbral de duda", tau, "", f"cuantil {tam['duda']} de la certeza")
    out("suma", variante, "abstenciones", th.abstained, "", "respuestas de las que no aprendió")
    out("suma", variante, "créditos a la percepción", st.get("creditos", 0), "", "")
    if st.get("creditos"):
        out("suma", variante, "precisión del crédito", st["creditos_bien"] / st["creditos"], "",
            "símbolo enseñado == dígito real")
    for n in (1, 3):
        m = evaluar_suma(th, per, datos, rng, n, tam["ep_eval"])
        cota = ei.cota_independiente(p, 2 * n, ref[n])
        out("suma", variante, f"exacta n={n}", m["exacta"], cota, "cota indep. p^2n*oráculo")
        out("suma", variante, f"columna n={n}", m["columna"], "", "")
        out("suma", variante, f"todas las imágenes bien n={n}", m["todas_bien"], p ** (2 * n), "p^2n")
        out("suma", variante, f"exacta | todas bien n={n}", m["exacta_si_todas_bien"], ref[n], "oráculo")
    # ---- pruebas para romperlo (con lo ya entrenado, sin aprender)
    casos = [(f"ruido sigma={s}", 3, ruido(s, np.random.default_rng(seed + 77)), datos) for s in tam["sigmas"]]
    tipo = digitos_tipograficos(tam["n_tipo"], 900 + seed)
    if tipo is not None:
        casos.append(("estilo no visto (tipográficos)", 3, None, _Pool(*tipo)))
    casos += [(f"más columnas n={n}", n, None, datos) for n in tam["n_largos"] if n > 4]
    for nombre, n, tr, pool in casos:
        m = evaluar_suma(th, per, pool, rng, n, tam["ep_romper"], tr)
        cota = ei.cota_independiente(m["imagen"], 2 * n, ref[n])
        out("romper", f"{variante} | {nombre}", "exactitud imagen", m["imagen"], "", "")
        out("romper", f"{variante} | {nombre}", "exacta", m["exacta"], cota, "cota indep. p^2n*oráculo(n)")
        out("romper", f"{variante} | {nombre}", "exacta | todas bien", m["exacta_si_todas_bien"], ref[n], "oráculo(n)")


# ====================================================================== acarreo
def prueba_acarreo(seed, tam, out, variante=None):
    """Suma decimal con símbolos perfectos (oráculo): registros y cómo empiezan. Variante
    "largo": el doble de episodios (12000) para las configuraciones que importan."""
    datos = ei.Digitos(tam["n_pre"])
    largo = variante == "largo"
    configs = ((1, "cero"), (1, "escrito"), (2, "cero")) if largo else \
        ((1, "cero"), (1, "escrito"), (2, "cero"), (2, "escrito"))
    ep = tam["ep_suma"] * (2 if largo else 1)
    for R, reset in configs:
        rng = np.random.default_rng(1000 + seed)
        th = thinker_suma(seed, R, reset)
        t0 = time.time()
        entrenar_suma(th, Oraculo(), datos, rng, ep, {}, {})
        var = f"R={R} {reset}" + (f" ({ep} episodios)" if largo else "")
        for n in (1, 3, 8):
            m = evaluar_suma(th, Oraculo(), datos, rng, n, tam["ep_eval"])
            out("acarreo", var, f"exacta n={n}", m["exacta"], 10.0 ** -(n + 1), "azar")
            out("acarreo", var, f"columna n={n}", m["columna"], 0.1, "azar")
        out("acarreo", var, "segundos", round(time.time() - t0, 1), "", "")


# ====================================================================== escalera y L3
# credU: todas las alternativas, basta una sola explicación (explain="unica");
# cred2: sólo la segunda opción de la percepción, con las demás explicaciones descartadas
# por conocimiento establecido (explain="refutadas", el valor por defecto del Thinker)
VARIANTES_CONJ = {"base": dict(h=False, cred=False, k=10), "H": dict(h=True, cred=False, k=10),
                  "credU+H": dict(h=True, cred=True, k=10, explain="unica"),
                  "cred2+H": dict(h=True, cred=True, k=2), "cred+H": dict(h=True, cred=True, k=10)}


def episodio_escalera(nivel, th, per, datos, split, rng, learn, cfg, st):
    """Una columna de L0-L2 (ver exp_integration.py) con la interfaz nueva. La percepción
    recibe +-1 por el acierto (R-STDP, como en la ronda 1) y, con `cred`, el símbolo que
    explicó el error."""
    a, b = int(rng.integers(10)), int(rng.integers(10))
    ia = datos.una(split, a, rng)
    sa, _, aa = per.ver(ia, a, learn)
    imgs, sims, digs, alts = [ia], [sa], [a], [aa]
    th.reset()
    if nivel == "L0":
        y = a; tok = (sa,)
    elif nivel == "L1":
        y = (a + b) % 10; tok = (sa, 10 + b)
        alts.append([])                                      # b es un símbolo perfecto
    else:
        ib = datos.una(split, b, rng)
        sb, _, ab = per.ver(ib, b, learn)
        imgs.append(ib); sims.append(sb); digs.append(b); alts.append(_desplazar(ab, 10))
        y = (a + b) % 10; tok = (sa, 10 + sb)
    out = th.answer(tok, explore=learn)
    if learn:
        th.feedback(target=y, alternatives=alts if cfg["cred"] else None)
        _credito(th, per, imgs, (0, 10), digs, st)
        if hasattr(per, "reforzar"):
            for img, s in zip(imgs, sims):
                per.reforzar(img, s, 1.0 if out == y else -1.0)
    return out == y


def prueba_conjunta(seed, tam, out, variante, niveles):
    """Percepción y Thinker desde cero; única señal: el dígito de la columna."""
    cfg = VARIANTES_CONJ[variante]
    rng = np.random.default_rng(2000 + seed)
    datos = ei.Digitos(tam["n_pre"])
    Xt, yt = ei.muestra_test(datos, rng, tam["n_test_pureza"])
    for nivel in niveles:
        v = ei.Vista(seed, homeostasis=tam["homeostasis"] if cfg["h"] else 0.0)
        per = Percepcion(v, k=cfg["k"])
        th = Thinker(10 if nivel == "L0" else (20 if nivel in ("L1", "L2") else 21),
                     registers=2 if nivel == "L3" else 0, seed=seed,
                     explain=cfg.get("explain", "refutadas"))
        st: dict = {}
        scfg = dict(cred=cfg["cred"], refuerzo=True)
        t0 = time.time(); win = []
        for e in range(tam["ep_conjunta"]):
            if nivel == "L3":
                win += episodio_suma(th, per, datos, "train", rng, int(rng.integers(1, 5)), True, scfg, st)["columnas"]
            else:
                win.append(episodio_escalera(nivel, th, per, datos, "train", rng, True, cfg, st))
            if (e + 1) % tam["cada"] == 0:
                conf = v.confusion(Xt[:500], yt[:500])
                out("conjunta", f"{nivel} {variante}", f"curva columna ep={e + 1}", float(np.mean(win)), 0.1, "azar")
                out("conjunta", f"{nivel} {variante}", f"curva pureza ep={e + 1}", ei.pureza(conf), 0.1,
                    f"símbolos usados={int((conf.sum(0) > 0).sum())}")
                win = []
        conf = v.confusion(Xt, yt)
        pz = ei.pureza(conf)
        var = f"{nivel} {variante}"
        out("conjunta", var, "pureza percepción (test)", pz, 0.1, "azar")
        out("conjunta", var, "símbolos distintos usados", int((conf.sum(0) > 0).sum()), 10, "")
        out("conjunta", var, "créditos a la percepción", st.get("creditos", 0), "", "")
        if st.get("pares"):
            from scipy.optimize import linear_sum_assignment
            r, c = linear_sum_assignment(-conf)
            codigo = dict(zip(r.tolist(), c.tolist()))       # dígito -> símbolo de la percepción al final
            ok = [codigo[d] == s for d, s in st["pares"]]
            out("conjunta", var, "precisión del crédito (código final)", float(np.mean(ok)), 0.1,
                "símbolo enseñado == símbolo final del dígito real")
            mitad = len(ok) // 2
            out("conjunta", var, "precisión del crédito, 2.a mitad", float(np.mean(ok[mitad:])), 0.1, "")
        ei.plasticidad(th, False)
        if nivel == "L3":
            m = evaluar_suma(th, Percepcion(v), datos, rng, 3, tam["ep_eval"])
            out("conjunta", var, "exacta n=3 (test)", m["exacta"], ei.cota_independiente(pz, 6, 1.0),
                "pureza^6 (Thinker perfecto)")
            out("conjunta", var, "columna n=3 (test)", m["columna"], 0.1, "azar")
        else:
            acc = float(np.mean([episodio_escalera(nivel, th, per, datos, "test", rng, False, cfg, {})
                                 for _ in range(tam["ep_eval"])]))
            k = 2 if nivel == "L2" else 1
            out("conjunta", var, "acierto tarea (test)", acc, ei.cota_independiente(pz, k, 1.0),
                f"pureza^{k} (cota si el Thinker fuera perfecto)")
        out("conjunta", var, "segundos", round(time.time() - t0, 1), "", "")


# ====================================================================== ligadura
VARIANTES_LIG = {"base": dict(), "conf": dict(conf="p"), "confcos": dict(conf="pcos"),
                 "cred": dict(cred=True), "cred2": dict(cred=True, k=2),
                 "confcos+cred2": dict(conf="pcos", cred=True, k=2)}


def episodio_ligadura(th, per, datos, split, rng, learn, cfg, st, ligada=None, F=None):
    """Como en la ronda 1 (4 pares clave-valor de imágenes y una pregunta con otra imagen),
    con la certeza de cada imagen en la memoria episódica. Con `cred`, las alternativas de
    las 9 imágenes entran en la explicación del error (la pregunta y lo ligado antes): si
    sólo se consideraba la pregunta, un valor mal percibido al ligar quedaba «explicado»
    por la pregunta (el dato permanente que da ese valor) y el crédito era falso (precisión
    0.41, la vista débil cayó de 0.88 a 0.63; semilla 0)."""
    claves = rng.permutation(10)[:5]
    valores = rng.integers(10, size=4)
    conf = bool(cfg.get("conf")); cred = bool(cfg.get("cred"))
    th.reset()
    ok_perc = True
    vistas = {}                                          # observación -> (imagen, desplazamiento, dígito)
    for k, val in zip(claves[:4], valores):
        ik, iv = datos.una(split, k, rng), datos.una(split, val, rng)
        sk, ck, ak = per.ver(ik, k, learn)
        sv, cv, av = per.ver(iv, val, learn)
        vistas[len(th._obs)] = (ik, 0, k)
        th.observe(sk, certainty=ck if conf else 1.0, alternatives=[ak] if cred else None)
        vistas[len(th._obs)] = (iv, 10, val)
        th.observe(10 + sv, certainty=cv if conf else 1.0, alternatives=[_desplazar(av, 10)] if cred else None)
        th.observe(None)
        ok_perc &= (sk == k) and (sv == val)
    ligada = rng.random() < 0.5 if ligada is None else ligada
    j = int(rng.integers(4))
    q = int(claves[j]) if ligada else int(claves[4])
    y = 10 + (int(valores[j]) if ligada else int(F[q]))
    iq = datos.una(split, q, rng)
    sq, cq, aq = per.ver(iq, q, learn)
    ok_perc &= sq == q
    out = th.answer(sq, explore=learn, certainty=cq if conf else 1.0)
    if learn:
        th.feedback(target=y, alternatives=[aq] if cred else None)
        if th.input_credit and isinstance(th.input_credit[0][0], tuple):   # culpa a algo ligado antes
            (t, _), sym = th.input_credit[0]
            img, off, dig = vistas[t]
            th.input_credit = [(0, sym)]
            _credito(th, per, (img,), (off,), (dig,), st)
        else:
            _credito(th, per, (iq,), (0,), (q,), st)
    return out == y, bool(ok_perc)


def evaluar_ligadura(th, per, datos, rng, F, episodios):
    ei.plasticidad(th, False)
    res = {}
    for lig in (True, False):
        antes = dict(th.stats())
        r = [episodio_ligadura(th, per, datos, "test", rng, False, {}, {}, lig, F) for _ in range(episodios)]
        hit = np.array([x[0] for x in r]); ok = np.array([x[1] for x in r])
        res[lig] = dict(acierto=float(hit.mean()),
                        si_bien=float(hit[ok].mean()) if ok.any() else float("nan"),
                        copiar=ei._fraccion(antes, th.stats(), "copiar"))
    ei.plasticidad(th, True)
    return res


def prueba_ligadura(seed, tam, out, variante=None):
    rng = np.random.default_rng(3000 + seed)
    F = np.random.default_rng(7).permutation(10)
    datos = ei.Digitos(tam["n_pre"])
    th_o = Thinker(20, memory=True, seed=seed)
    for _ in range(tam["ep_ligadura"]):
        episodio_ligadura(th_o, Oraculo(), datos, "train", rng, True, {}, {}, None, F)
    ref = evaluar_ligadura(th_o, Oraculo(), datos, rng, F, tam["ep_eval"])
    for lig, name in ((True, "ligadas"), (False, "no ligadas")):
        out("ligadura", "oraculo", f"acierto {name}", ref[lig]["acierto"], 0.1, "azar")
    Xt, yt = ei.muestra_test(datos, rng, tam["n_test"])
    cal = datos.X["train"][rng.choice(len(datos.y["train"]), tam["n_cal"], replace=False)]
    base = ei.vista_entrenada(datos, seed, tam["n_debil"], "depth1")
    p0 = float(np.trace(base.confusion(Xt, yt)) / len(yt))
    out("ligadura", "vista débil", "exactitud imagen (test)", p0, "", "depth1, 1000 etiquetas")
    for var in (VARIANTES_LIG if variante is None else [variante]):
        cfg = VARIANTES_LIG[var]
        v = copy.deepcopy(base)
        per = Percepcion(v, k=cfg.get("k", 10), certeza=cfg.get("conf") or "p")
        tau = per.umbral(cal, tam["duda"]) if cfg.get("conf") else 0.0
        th = Thinker(20, memory=True, seed=seed, doubt=tau)
        st: dict = {}
        t0 = time.time()
        for _ in range(tam["ep_ligadura"]):
            episodio_ligadura(th, per, datos, "train", rng, True, cfg, st, None, F)
        p = float(np.trace(v.confusion(Xt, yt)) / len(yt))
        out("ligadura", var, "vista débil después (test)", p, p0, "antes")
        out("ligadura", var, "abstenciones", th.abstained, "", f"umbral {tau:.3f}")
        out("ligadura", var, "créditos a la percepción", st.get("creditos", 0), "", "")
        if st.get("creditos"):
            out("ligadura", var, "precisión del crédito", st["creditos_bien"] / st["creditos"], "", "")
        res = evaluar_ligadura(th, per, datos, rng, F, tam["ep_eval"])
        for lig, name, k in ((True, "ligadas", 3), (False, "no ligadas", 1)):
            m = res[lig]
            out("ligadura", var, f"acierto {name}", m["acierto"],
                ei.cota_independiente(p, k, ref[lig]["acierto"]), f"p^{k}*oráculo (p después)")
            out("ligadura", var, f"acierto {name} | imágenes bien", m["si_bien"], ref[lig]["acierto"], "oráculo")
            out("ligadura", var, f"fracción COPIAR {name}", m["copiar"], "", "")
        out("ligadura", var, "segundos", round(time.time() - t0, 1), "", "")


# ====================================================================== 4b
VARIANTES_INTERF = {"control: sigue con oráculo": None, "base": {}, "conf": dict(conf="p"),
                    "confcos": dict(conf="pcos"), "cred2": dict(cred=True, k=2),
                    "confcos+cred2": dict(conf="pcos", cred=True, k=2)}


def prueba_interf(seed, tam, out, variante=None):
    rng = np.random.default_rng(6000 + seed)
    datos = ei.Digitos(tam["n_pre"])
    cal = datos.X["train"][rng.choice(len(datos.y["train"]), tam["n_cal"], replace=False)]
    debil = ei.vista_entrenada(datos, seed, tam["n_debil"], "depth1")
    th0 = thinker_suma(seed)
    entrenar_suma(th0, Oraculo(), datos, rng, tam["ep_suma"], {}, {})
    m0 = evaluar_suma(th0, Oraculo(), datos, rng, 3, tam["ep_eval"])
    out("interf", "4b", "Thinker antes: exacta n=3 con oráculo", m0["exacta"], "", "")
    Xt, yt = ei.muestra_test(datos, np.random.default_rng(60 + seed), tam["n_test_interf"])
    p0 = float(np.trace(debil.confusion(Xt, yt)) / len(yt))
    out("interf", "4b", "vista débil (test)", p0, "", "depth1")
    for var, cfg in VARIANTES_INTERF.items():
        th = copy.deepcopy(th0)
        st: dict = {}
        if cfg is None:
            entrenar_suma(th, Oraculo(), datos, rng, tam["ep_interf"], {}, st)
        else:
            per = Percepcion(copy.deepcopy(debil), k=cfg.get("k", 10), certeza=cfg.get("conf") or "p")
            th.doubt = per.umbral(cal, tam["duda"]) if cfg.get("conf") else 0.0
            entrenar_suma(th, per, datos, rng, tam["ep_interf"], dict(conf=bool(cfg.get("conf")), cred=cfg.get("cred")), st)
        m = evaluar_suma(th, Oraculo(), datos, rng, 3, tam["ep_eval"])
        out("interf", f"4b {var}", "Thinker después: exacta n=3 con oráculo", m["exacta"], m0["exacta"], "antes")
        out("interf", f"4b {var}", "Thinker después: columna n=3 con oráculo", m["columna"], m0["columna"], "antes")
        out("interf", f"4b {var}", "abstenciones", th.abstained, "", "")
        out("interf", f"4b {var}", "créditos a la percepción", st.get("creditos", 0), "", "")
        if st.get("creditos"):
            out("interf", f"4b {var}", "precisión del crédito", st["creditos_bien"] / st["creditos"], "",
                "símbolo enseñado == dígito real")
            out("interf", f"4b {var}", "vista débil después (test)",
                float(np.trace(per.v.confusion(Xt, yt)) / len(yt)), p0, "antes")


# ====================================================================== adaptación a un estilo no visto
VARIANTES_ESTILO = {"control: nadie aprende": None, "refuerzo (ronda 1)": dict(refuerzo=True),
                    "cred2": dict(cred=True, k=2), "cred": dict(cred=True, k=10)}


def prueba_estilo(seed, tam, out, variante=None):
    """¿El razonamiento puede enseñarle a la percepción un estilo que nunca vio, sin
    etiquetas? Percepción entrenada con MNIST manuscrito y Thinker que ya sabe sumar
    (con símbolos perfectos) reciben sumas cuyos dígitos son tipográficos. La única señal
    es el dígito de cada columna. Se mide la percepción sobre otros ejemplares
    tipográficos (mismas 10 tipografías, otra rotación y tamaño), sobre MNIST (olvido) y
    la suma con símbolos perfectos (si el Thinker se degradó)."""
    rng = np.random.default_rng(8000 + seed)
    datos = ei.Digitos(tam["n_pre"])
    tipo_tr = digitos_tipograficos(tam["n_tipo_tr"], 800 + seed)
    tipo_te = digitos_tipograficos(tam["n_tipo"], 900 + seed)
    if tipo_tr is None:
        return
    pool = _Pool(*tipo_tr)
    Xt, yt = ei.muestra_test(datos, rng, tam["n_test_interf"])
    base = ei.vista_entrenada(datos, seed, tam["n_pre"])
    th0 = thinker_suma(seed)
    entrenar_suma(th0, Oraculo(), datos, rng, tam["ep_suma"], {}, {})
    m0 = evaluar_suma(th0, Oraculo(), datos, rng, 3, tam["ep_eval"])
    acc = lambda v, X, y: float(np.trace(v.confusion(X, y)) / len(y))  # noqa: E731
    a_tipo0, a_mnist0 = acc(base, *tipo_te), acc(base, Xt, yt)
    out("estilo", "antes", "vista: tipográficos (test)", a_tipo0, "", "nunca vio este estilo")
    out("estilo", "antes", "vista: MNIST (test)", a_mnist0, "", "")
    out("estilo", "antes", "suma exacta n=3 con oráculo", m0["exacta"], "", "")
    for var, cfg in VARIANTES_ESTILO.items():
        v = copy.deepcopy(base)
        th = copy.deepcopy(th0)
        per = Percepcion(v, k=(cfg or {}).get("k", 2))
        st: dict = {}
        t0 = time.time()
        if cfg is not None:
            entrenar_suma(th, per, pool, rng, tam["ep_estilo"], cfg, st)
        out("estilo", var, "vista: tipográficos (test)", acc(v, *tipo_te), a_tipo0, "antes")
        out("estilo", var, "vista: MNIST (test)", acc(v, Xt, yt), a_mnist0, "antes")
        m = evaluar_suma(th, per, _Pool(*tipo_te), rng, 3, tam["ep_romper"])
        out("estilo", var, "suma exacta n=3 con tipográficos (test)", m["exacta"],
            ei.cota_independiente(m["imagen"], 6, m0["exacta"]), "cota indep. p^6*oráculo")
        m = evaluar_suma(th, Oraculo(), datos, rng, 3, tam["ep_eval"])
        out("estilo", var, "suma exacta n=3 con oráculo", m["exacta"], m0["exacta"], "antes")
        out("estilo", var, "créditos a la percepción", st.get("creditos", 0), "", "")
        if st.get("creditos"):
            out("estilo", var, "precisión del crédito", st["creditos_bien"] / st["creditos"], "", "")
        out("estilo", var, "segundos", round(time.time() - t0, 1), "", "")


# ====================================================================== principal
TAM = dict(
    completa=dict(n_pre=10000, n_debil=1000, n_test=5000, n_test_pureza=2000, ep_suma=6000, ep_eval=500,
                  ep_conjunta=4000, cada=1000, ep_ligadura=3000, ep_interf=2000, homeostasis=0.02,
                  duda=0.2, n_cal=500, sigmas=(0.5, 0.7), n_tipo=60, ep_romper=300, n_largos=(1, 3, 6, 10),
                  n_tipo_tr=200, ep_estilo=1500, n_test_interf=2000),
    rapida=dict(n_pre=300, n_debil=200, n_test=200, n_test_pureza=100, ep_suma=40, ep_eval=10,
                ep_conjunta=20, cada=10, ep_ligadura=30, ep_interf=20, homeostasis=0.02,
                duda=0.2, n_cal=50, sigmas=(0.5,), n_tipo=3, ep_romper=5, n_largos=(1, 3, 6),
                n_tipo_tr=3, ep_estilo=10, n_test_interf=50),
)
PRUEBAS = dict(
    acarreo=prueba_acarreo,
    suma=prueba_suma,
    escalera=lambda s, t, o, v: prueba_conjunta(s, t, o, v, ("L0", "L1", "L2")),
    conjunta=lambda s, t, o, v: prueba_conjunta(s, t, o, v, ("L3",)),
    ligadura=prueba_ligadura,
    interf=prueba_interf,
    estilo=prueba_estilo,
)
DEFECTO = dict(suma="base", escalera="base", conjunta="base")


def resumen():
    """Media, mínimo y máximo por (prueba, variante, métrica) sobre las semillas."""
    filas = {}
    for p in sorted(RESULTS.glob("integracion2_*_seed*.csv")):
        with p.open() as f:
            for r in csv.DictReader(f):
                try:
                    v = float(r["valor"])
                except ValueError:
                    continue
                k = (r["prueba"], r["variante"], r["metrica"])
                filas.setdefault(k, dict(v=[], s=[], ref=[], nota=r["nota"]))
                filas[k]["v"].append(v); filas[k]["s"].append(r["semilla"])
                try:
                    filas[k]["ref"].append(float(r["referencia"]))
                except ValueError:
                    pass
    log = CsvLog("integracion2_resumen", ["prueba", "variante", "metrica", "media", "min", "max",
                                          "semillas", "por_semilla", "referencia_media", "nota"])
    for (pr, var, met), d in sorted(filas.items()):
        v = np.array(d["v"])
        log.row(pr, var, met, float(v.mean()), float(v.min()), float(v.max()), len(v),
                " ".join(f"{x:.3f}" for x in v),
                float(np.mean(d["ref"])) if d["ref"] else "", d["nota"])
    log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prueba", choices=list(PRUEBAS) + ["resumen"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variante", default=None)
    ap.add_argument("--rapida", action="store_true", help="tamaños mínimos (humo), sólo imprime")
    args = ap.parse_args()
    if args.prueba == "resumen":
        resumen()
        return
    var = args.variante or DEFECTO.get(args.prueba)
    t0 = time.time()
    if args.rapida:
        PRUEBAS[args.prueba](args.seed, TAM["rapida"], lambda *r: print(*r, sep="  ", flush=True), var)
        print("segundos", round(time.time() - t0, 1))
        return
    nombre = f"integracion2_{args.prueba}" + (f"_{var.replace('+', '-')}" if var else "") + f"_seed{args.seed}"
    log = CsvLog(nombre, HEADER)
    PRUEBAS[args.prueba](args.seed, TAM["completa"], lambda *r: log.row(r[0], r[1], args.seed, *r[2:]), var)
    log.row(args.prueba, "total", args.seed, "segundos", round(time.time() - t0, 1), "", "")
    log.close()


if __name__ == "__main__":
    main()
