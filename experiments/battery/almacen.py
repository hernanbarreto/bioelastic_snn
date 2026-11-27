"""Resultados crudos por prueba y semilla: results/bateria/<commit>/<prueba>__<variante>/s<k>.json.

Cada semilla se escribe apenas termina (escritura atómica: archivo temporal y rename) y se
salta si ya existe con la misma versión del adaptador. Así una corrida cortada se retoma
donde quedó.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results"


class Almacen:
    def __init__(self, commit: str, raiz: Path | str | None = None):
        self.commit = commit
        self.dir = Path(raiz or RESULTS) / "bateria" / commit

    def ruta(self, clave: str, semilla: int) -> Path:
        return self.dir / clave / f"s{int(semilla)}.json"

    def cargar(self, clave: str, semilla: int, version: int, escala: float | None = None) -> dict | None:
        """El resultado guardado, si existe con la misma versión del adaptador (y la misma
        escala, si se pide)."""
        p = self.ruta(clave, semilla)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None                               # archivo cortado: se vuelve a correr
        if d.get("version") != version:
            return None
        if escala is not None and abs(float(d.get("escala", -1.0)) - float(escala)) > 1e-12:
            return None
        return d

    def guardar(self, clave: str, semilla: int, crudo: dict) -> Path:
        p = self.ruta(clave, semilla)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(crudo, ensure_ascii=False, indent=1, default=_jsonable) + "\n",
                       encoding="utf-8")
        os.replace(tmp, p)
        return p


def _jsonable(x):
    import numpy as np
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x))
