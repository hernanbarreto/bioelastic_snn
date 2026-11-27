"""Regresión rápida de `experiments/exp_thinking_hard` con los valores por defecto (una semilla):
lo que la v0.3 ya resolvía no puede dejar de resolverse. (C1 con vocabulario 256 cayó de 0.999 a
0.02 con todo encendido sin que ningún test lo viera: la contracción del control no iba con la
ganancia de su plasticidad y el ruido a priori de la memoria pedía una repetición más para
consolidar.) Umbrales: lo que da la v0.3 en esa semilla, con margen."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import exp_thinking_hard as H  # noqa: E402

W32, W256, W64 = H.World(32), H.World(256), H.World(64)
TASKS = (0, 1, 2, 3, 4, 5, 6)
CASES = [
    # v0.3 con este código, 3 semillas: 0.835 +- 0.035 / 0.888 +- 0.012; valores por defecto: 0.805 +- 0.011 /
    # 0.835 +- 0.035 (una semilla, 0.795). El umbral es para lo catastrófico (cayó a 0.02).
    ("C1 vocabulario 256", W256, 2, {}, H.plan_single(1, 3000), lambda: H.evals_for(W256, 1, cue=1), 0.75),
    ("C3 vocabulario 256", W256, 1, {}, H.plan_single(3, 3000), lambda: H.evals_for(W256, 3, cue=0), 0.95),
    ("C5 vocabulario 256", W256, 1, {}, H.plan_single(5, 3000), lambda: H.evals_for(W256, 5, cue=0), 0.95),
    ("C3 cadenas largas", W64, 1, dict(max_steps=20), H.plan_single(3, 3000),
     lambda: [("12-16 saltos", lambda net, r: H.c3(W64, net, r, False, 0, hops=int(r.integers(12, 17)), distractors=8))], 0.95),
    pytest.param("multitarea en bloques, orden inverso", W32, 7, {}, H.plan_blocks(TASKS, 1500, 750, reverse=True),
                 lambda: [(f"C{t} {name}", fn) for t in TASKS for name, fn in H.evals_for(W32, t, t)], 0.9,
                 marks=pytest.mark.xfail(strict=True, reason=(
                     "regresión abierta: v0.3 1.00 en todo (3 semillas); valores por defecto C1 0.517, C4 0.683, C2 0.777, "
                     "C0 0.852 (f aprendida al final: lo de las composiciones no se repara con la repetición)"))),
]


@pytest.mark.parametrize("name,W,cues,kw,plan,evals,low", CASES,
                         ids=[(c.values[0] if hasattr(c, "values") else c[0]) for c in CASES])
def test_hard_case_with_defaults(name, W, cues, kw, plan, evals, low):
    out, _ = H.run(W, lambda s: H.make_thinker(W, cues, seed=s, **kw), plan, evals(), (0,), 200)
    worst = min(float(np.mean(v)) for v in out.values())
    assert worst >= low, {k: round(float(np.mean(v)), 3) for k, v in out.items()}
