"""Utilidades comunes de los experimentos."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))


class CsvLog:
    """CSV incremental en results/<name>.csv; además imprime cada fila."""

    def __init__(self, name: str, header: list[str]):
        RESULTS.mkdir(exist_ok=True)
        self.path = RESULTS / f"{name}.csv"
        self.header = header
        self._f = self.path.open("w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(header)

    def row(self, *vals):
        out = [f"{v:.4f}" if isinstance(v, float) else v for v in vals]
        self._w.writerow(out); self._f.flush()
        print("  ".join(f"{h}={v}" for h, v in zip(self.header, out)), flush=True)

    def close(self):
        self._f.close()
