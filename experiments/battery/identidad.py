"""Comprobación de identidad bit a bit de la percepción entre dos commits.

  python experiments/battery/identidad.py <commit> <salida.npz>     (un commit por proceso)
  python experiments/battery/identidad.py comparar <a.npz> <b.npz>
  python experiments/battery/identidad.py equivalencia <a.npz> <b.npz>   (escribe la equivalencia)

Con la configuración por defecto, sobre una muestra: la red depth2 entrenada con 1 000
imágenes de MNIST (sus decisiones y potenciales en 300 del test), V1 sin etiquetas con 500
imágenes de Omniglot (los códigos de 20) y la red de texto con 5 000 bytes (sus decisiones y
probabilidades en 1 000 bytes siguientes). Si todo es idéntico bit a bit, la batería puede
reusar entre esos commits la cache de percepción y los resultados de las pruebas que no
usan el Thinker (lo declara en results/bateria/equivalencias.json).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def muestra(commit: str, salida: str):
    from battery import version
    info = version.preparar(commit)
    from bioelastic import BioElasticNet, Cortex, FeatureArea, TextEncoder
    from battery import lenguaje, omniglot, percepcion
    out = {}
    Xtr, ytr, Xte, _ = percepcion.mnist()
    E = percepcion.encoder()
    net = Cortex((28, 28, 1), 10, seed=0, **percepcion.DEPTH2)
    for i in np.random.default_rng(0).permutation(len(Xtr) - 5000)[:1000]:
        net.step(E(Xtr[i]), int(ytr[i]))
    for a in net.areas:
        a._cache = None
    pred, pot = [], []
    for x in Xte[:300]:
        pred.append(net.forward(E(x))); pot.append(np.asarray(net._V, dtype=np.float32).copy())
    out["cortex_pred"] = np.array(pred); out["cortex_V"] = np.stack(pot)
    X, _, _ = omniglot.cargar_fondo()
    area = FeatureArea((28, 28, 1), seed=100, **omniglot.V1)
    Eo = omniglot._encoder()
    for i in range(500):
        area.encode(Eo(X[i])); area.learn(None)
    out["v1_codigos"] = np.stack([area.encode(Eo(X[i])) for i in range(500, 520)])
    data = TextEncoder()(lenguaje.CORPUS.read_bytes()[:6001])
    tn = BioElasticNet(256, 256, seed=0, **lenguaje.CFG)
    for t in range(5000):
        tn.forward(int(data[t])); tn.teach(int(data[t + 1]))
    tp, pp = [], []
    for t in range(5000, 6000):
        tp.append(tn.forward(int(data[t]))); pp.append(tn.readout.probs(tn._last.V).astype(np.float64))
    out["texto_pred"] = np.array(tp); out["texto_p"] = np.stack(pp)
    np.savez(salida, commit=np.array(info["sha"]), **out)


def comparar(a: str, b: str) -> dict:
    za, zb = np.load(a), np.load(b)
    res = {}
    for k in za.files:
        if k == "commit":
            continue
        x, y = za[k], zb[k]
        res[k] = bool(x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes())
    return dict(identico=all(res.values()), por_parte=res, commits=[str(za["commit"]), str(zb["commit"])])


def equivalencia(a_npz: str, b_npz: str) -> dict:
    """Si las dos muestras son idénticas bit a bit y las huellas de percepción difieren,
    registra en results/bateria/equivalencias.json que la del segundo commit equivale a la
    del primero (el informe hereda entonces las pruebas de percepción con opciones por
    defecto)."""
    import json
    import time
    from battery import version
    r = comparar(a_npz, b_npz)
    ca, cb = r["commits"]
    ha, hb = version.huella_percepcion(ca), version.huella_percepcion(cb)
    r.update(huellas=[ha, hb], escrita=False)
    if r["identico"] and ha != hb:
        eq = json.loads(version.EQUIVALENCIAS.read_text()) if version.EQUIVALENCIAS.exists() else {}
        eq.setdefault("percepcion", {})[hb] = dict(
            igual_a=ha, commits=[ca, cb], comprobacion=r["por_parte"], fecha=time.strftime("%Y-%m-%d %H:%M"),
            metodo="battery/identidad.py: depth2 con 1 000 imágenes de MNIST (decisiones y potenciales en 300 del "
                   "test), V1 con 500 de Omniglot (códigos de 20) y red de texto con 5 000 bytes (decisiones y "
                   "probabilidades en 1 000), configuración por defecto, idénticos bit a bit")
        version.EQUIVALENCIAS.parent.mkdir(parents=True, exist_ok=True)
        version.EQUIVALENCIAS.write_text(json.dumps(eq, indent=1, ensure_ascii=False) + "\n")
        r["escrita"] = True
    return r


if __name__ == "__main__":
    import json
    if sys.argv[1] == "comparar":
        print(json.dumps(comparar(sys.argv[2], sys.argv[3]), indent=1))
    elif sys.argv[1] == "equivalencia":
        print(json.dumps(equivalencia(sys.argv[2], sys.argv[3]), indent=1))
    else:
        muestra(sys.argv[1], sys.argv[2])
