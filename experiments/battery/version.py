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


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def resolver(ref: str = "HEAD") -> str:
    return _git("rev-parse", "--verify", f"{ref}^{{commit}}")


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
