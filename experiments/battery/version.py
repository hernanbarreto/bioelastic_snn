"""Qué versión de la red se mide.

La batería vive en esta rama, pero puede medir el paquete `bioelastic/` de cualquier
commit del repositorio: lo extrae con `git archive` a ext_data/commits/<sha>/ y lo pone
primero en `sys.path`. Así se repite la misma batería después de cada fusión y se sigue el
índice por versión. Si el commit es el HEAD de este árbol y `bioelastic/` no tiene cambios,
se usa el árbol tal cual.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "ext_data" / "commits"
# Archivos de los que depende la percepción (Cortex y sus piezas, también la red de texto).
# Si son idénticos entre dos commits, las pruebas que no usan el Thinker dan lo mismo (el
# entrenamiento es determinista) y el informe puede heredar sus resultados.
ARCHIVOS_PERCEPCION = ("__init__.py", "area.py", "cortex.py", "datasets.py", "encoders.py", "layer.py",
                       "network.py", "neuron.py", "readout.py")


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def resolver(ref: str = "HEAD") -> str:
    return _git("rev-parse", "--verify", f"{ref}^{{commit}}")


def arbol_bioelastic(ref: str) -> str:
    """El hash git del árbol bioelastic/ del commit: igual si y sólo si el código medido es
    idéntico byte a byte."""
    return _git("rev-parse", f"{resolver(ref)}:bioelastic")


def huella(contenidos) -> str:
    """Huella de los archivos de percepción a partir de (nombre, bytes)."""
    import hashlib
    h = hashlib.sha1()
    for nombre, datos in contenidos:
        h.update(nombre.encode() + b"\0" + (datos or b""))
    return "p" + h.hexdigest()[:10]


EQUIVALENCIAS = ROOT / "results" / "bateria" / "equivalencias.json"


def huella_equivalente(h: str) -> str:
    """La huella canónica: si una comprobación bit a bit (battery/identidad.py) mostró que el
    código de percepción con huella `h` da lo mismo que otro, se usa la de ese otro."""
    import json
    if not EQUIVALENCIAS.exists():
        return h
    eq = json.loads(EQUIVALENCIAS.read_text()).get("percepcion", {})
    vistos = set()
    while h in eq and h not in vistos:
        vistos.add(h); h = eq[h]["igual_a"]
    return h


def huella_percepcion(ref: str) -> str:
    """La huella de percepción de un commit, leída con git (sin importar nada)."""
    sha = resolver(ref)
    out = []
    for nombre in ARCHIVOS_PERCEPCION:
        r = subprocess.run(["git", "show", f"{sha}:bioelastic/{nombre}"], cwd=ROOT, capture_output=True)
        out.append((nombre, r.stdout if r.returncode == 0 else b""))
    return huella(out)


def preparar(ref: str = "HEAD") -> dict:
    """Deja importable el `bioelastic` del commit `ref`. Devuelve la etiqueta del commit
    (12 caracteres, con «+sucio» si se mide el árbol con cambios sin commitear), el sha y
    la raíz desde la que se importa."""
    sha = resolver(ref)
    head = resolver("HEAD")
    if sha == head:
        sucio = bool(_git("status", "--porcelain", "--", "bioelastic"))
        raiz = ROOT
    else:
        sucio = False
        raiz = CACHE / sha
        if not (raiz / "bioelastic" / "__init__.py").exists():
            raiz.mkdir(parents=True, exist_ok=True)
            arch = subprocess.run(["git", "archive", sha, "bioelastic"], cwd=ROOT, check=True,
                                  capture_output=True).stdout
            subprocess.run(["tar", "-x", "-C", str(raiz)], input=arch, check=True)
    mod = sys.modules.get("bioelastic")
    if mod is not None and not Path(mod.__file__).resolve().is_relative_to(raiz.resolve()):
        raise RuntimeError(f"bioelastic ya está importado desde {mod.__file__}; "
                           "preparar() tiene que llamarse antes de importarlo")
    sys.path.insert(0, str(raiz))
    etiqueta = sha[:12] + ("+sucio" if sucio else "")
    return dict(commit=etiqueta, sha=sha, raiz=str(raiz), sucio=sucio)
